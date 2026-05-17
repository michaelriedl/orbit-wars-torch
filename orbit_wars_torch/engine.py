"""GPU-native batched Orbit Wars engine.

Runs `B` envs in lockstep as a set of `(B, ...)` tensors on a torch
device. Map generation and comet path generation stay on CPU (they are
once-per-episode / once-per-spawn rejection samplers that don't
vectorize), but every per-step game tick -- planet rotation, fleet
movement, continuous swept-pair collision, combat resolution -- runs as
batched tensor ops.

Design notes:

- Planet buffer is fixed-size (`cfg.max_planets` + `comet_max_groups * 4`
  reserved comet slots). Unused slots carry `alive=False`.
- Fleet buffer is dynamic: starts at `max_fleets_initial`, doubles
  whenever the in-use count exceeds 90% of capacity.
- Comets store precomputed paths as `(B, max_groups, 4, max_path, 2)`
  tensors plus per-group `path_index` and `path_len`. New paths are
  generated on CPU and uploaded into the next free group slot.
- Swept-pair collision uses argmin(t1) per fleet across all planet
  endpoints; this differs from the upstream "first planet in iteration
  order" only when a fleet path intersects two planets in one tick (the
  earliest hit wins instead of the lowest planet id). In practice these
  collisions are extremely rare.

Public API:

- `TorchOrbitWarsEnv(cfg).reset(seeds)` -> initializes B envs.
- `.step(actions_p0, actions_p1)` -> 2-player shortcut for one tick.
- `.step_players(actions_per_player)` -> N-player tick.
- `.dump_states()` -> per-env JSON-friendly state dicts in the same shape
  as the Kaggle env's `dump_state`.

Single-file copy intended for Kaggle sharing; no project-internal imports.
"""

# B/P/F/G/Q/L are tensor-shape conventions (batch, planet-slots, fleet-slots,
# comet groups, quadrants, comet path length). Uppercase is the common ML
# convention; we suppress ruff's N806 check for this file only.
# ruff: noqa: N806

from __future__ import annotations

import os
import math
import random
import logging
import concurrent.futures
from typing import Any
from dataclasses import field, dataclass

import torch
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    CENTER,
    BOARD_SIZE,
    SUN_RADIUS,
    COMET_RADIUS,
    COMET_PRODUCTION,
    COMET_SPAWN_STEPS,
    ROTATION_RADIUS_LIMIT,
    generate_planets,
    generate_comet_paths,
)


def _comet_worker(args: tuple) -> tuple[list, int]:
    """Worker for the comet-spawn process pool. Returns (paths, ships).

    Top-level (not a method) so it pickles for ProcessPoolExecutor.
    """
    initial_planets, omega, spawn_step, live_ids, comet_speed, rng_key = args
    rng = random.Random(rng_key)
    paths = generate_comet_paths(
        initial_planets, omega, spawn_step, live_ids, comet_speed, rng=rng
    )
    if not paths:
        return [], 0
    comet_ships = min(
        rng.randint(1, 99),
        rng.randint(1, 99),
        rng.randint(1, 99),
        rng.randint(1, 99),
    )
    return paths, comet_ships


logger = logging.getLogger(__name__)

EMPTY_OWNER = -2  # Sentinel for "slot is empty"; -1 is neutral.
EMPTY_ID = -1


@dataclass
class TorchEngineConfig:
    """Per-batch sizing for the GPU engine."""

    num_envs: int
    num_agents: int = 2
    max_planets: int = 48
    comet_max_groups: int = 5
    comet_quadrants: int = 4
    comet_max_path: int = 40
    max_fleets_initial: int = 256
    episode_steps: int = 500
    ship_speed: float = 6.0
    comet_speed: float = 4.0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float32


class TorchOrbitWarsEnv:
    """B-batched Orbit Wars engine living on a single torch device."""

    def __init__(self, cfg: TorchEngineConfig) -> None:
        self.cfg = cfg
        self.B = cfg.num_envs
        self.P = cfg.max_planets + cfg.comet_max_groups * cfg.comet_quadrants
        self.F = cfg.max_fleets_initial
        self.G = cfg.comet_max_groups
        self.Q = cfg.comet_quadrants
        self.L = cfg.comet_max_path
        self.device = cfg.device
        self.dtype = cfg.dtype
        self._initial_planets_cpu: list[list[list[float]]] = [[] for _ in range(self.B)]
        self._comet_groups_cpu: list[list[dict[str, Any]]] = [[] for _ in range(self.B)]
        self._done_cpu: list[bool] = [False] * self.B
        self._step_count_cpu: list[int] = [0] * self.B
        self._rewards_cpu: list[list[int]] = [[0] * cfg.num_agents for _ in range(self.B)]
        self._comet_pool: concurrent.futures.ProcessPoolExecutor | None = None
        self._comet_pool_threshold = 8
        self._alloc_state()

    # State allocation ------------------------------------------------

    def _alloc_state(self) -> None:
        B, P, F, G, Q, L = self.B, self.P, self.F, self.G, self.Q, self.L
        d = self.device
        ft = self.dtype

        self.planet_owner = torch.full((B, P), EMPTY_OWNER, dtype=torch.int64, device=d)
        self.planet_id = torch.full((B, P), EMPTY_ID, dtype=torch.int64, device=d)
        self.planet_x = torch.zeros((B, P), dtype=ft, device=d)
        self.planet_y = torch.zeros((B, P), dtype=ft, device=d)
        self.planet_radius = torch.zeros((B, P), dtype=ft, device=d)
        self.planet_ships = torch.zeros((B, P), dtype=torch.int64, device=d)
        self.planet_production = torch.zeros((B, P), dtype=torch.int64, device=d)
        self.planet_alive = torch.zeros((B, P), dtype=torch.bool, device=d)
        self.planet_initial_x = torch.zeros((B, P), dtype=ft, device=d)
        self.planet_initial_y = torch.zeros((B, P), dtype=ft, device=d)
        self.planet_is_comet = torch.zeros((B, P), dtype=torch.bool, device=d)

        self.fleet_owner = torch.full((B, F), EMPTY_OWNER, dtype=torch.int64, device=d)
        self.fleet_id = torch.full((B, F), EMPTY_ID, dtype=torch.int64, device=d)
        self.fleet_x = torch.zeros((B, F), dtype=ft, device=d)
        self.fleet_y = torch.zeros((B, F), dtype=ft, device=d)
        self.fleet_angle = torch.zeros((B, F), dtype=ft, device=d)
        self.fleet_ships = torch.zeros((B, F), dtype=torch.int64, device=d)
        self.fleet_from_pid = torch.full((B, F), EMPTY_ID, dtype=torch.int64, device=d)
        self.fleet_alive = torch.zeros((B, F), dtype=torch.bool, device=d)

        self.comet_paths = torch.zeros((B, G, Q, L, 2), dtype=ft, device=d)
        self.comet_path_len = torch.zeros((B, G), dtype=torch.int64, device=d)
        self.comet_group_alive = torch.zeros((B, G), dtype=torch.bool, device=d)
        self.comet_group_path_idx = torch.full((B, G), -1, dtype=torch.int64, device=d)
        self.comet_group_planet_slot = torch.full((B, G, Q), -1, dtype=torch.int64, device=d)

        self.step_count = torch.zeros((B,), dtype=torch.int64, device=d)
        self.angular_velocity = torch.zeros((B,), dtype=ft, device=d)
        self.next_fleet_id = torch.zeros((B,), dtype=torch.int64, device=d)
        self.next_planet_id = torch.zeros((B,), dtype=torch.int64, device=d)
        self.next_comet_group_idx = torch.zeros((B,), dtype=torch.int64, device=d)
        self.next_planet_slot = torch.zeros((B,), dtype=torch.int64, device=d)
        self.done = torch.zeros((B,), dtype=torch.bool, device=d)
        self.rewards = torch.zeros((B, self.cfg.num_agents), dtype=torch.int64, device=d)
        self._episode_seeds = [0 for _ in range(self.B)]

    # Reset ----------------------------------------------------------

    def reset(self, seeds: list[int]) -> None:
        """Re-initialize every env with a fresh map drawn from `seeds[i]`."""

        if len(seeds) != self.B:
            raise ValueError(f"expected {self.B} seeds, got {len(seeds)}")
        for env_idx, seed in enumerate(seeds):
            self._reset_one(env_idx, int(seed))
        empty = [[[] for _ in range(self.B)] for _ in range(self.cfg.num_agents)]
        self.step_players(empty)

    def _reset_one(self, env_idx: int, seed: int) -> None:
        rng = random.Random(seed)
        angular_velocity = rng.uniform(0.025, 0.05)
        planets = generate_planets(rng)
        initial_planets = [p[:] for p in planets]

        num_groups = len(planets) // 4
        if num_groups > 0:
            home_group = rng.randint(0, num_groups - 1)
            base = home_group * 4
            if self.cfg.num_agents == 2:
                planets[base][1] = 0
                planets[base][5] = 10
                planets[base + 3][1] = 1
                planets[base + 3][5] = 10
            elif self.cfg.num_agents == 4:
                for j in range(4):
                    planets[base + j][1] = j
                    planets[base + j][5] = 10

        self.planet_owner[env_idx].fill_(EMPTY_OWNER)
        self.planet_id[env_idx].fill_(EMPTY_ID)
        self.planet_alive[env_idx].fill_(False)
        self.planet_is_comet[env_idx].fill_(False)
        self.planet_x[env_idx].zero_()
        self.planet_y[env_idx].zero_()
        self.planet_radius[env_idx].zero_()
        self.planet_ships[env_idx].zero_()
        self.planet_production[env_idx].zero_()
        self.planet_initial_x[env_idx].zero_()
        self.planet_initial_y[env_idx].zero_()

        self.fleet_owner[env_idx].fill_(EMPTY_OWNER)
        self.fleet_id[env_idx].fill_(EMPTY_ID)
        self.fleet_alive[env_idx].fill_(False)
        self.fleet_x[env_idx].zero_()
        self.fleet_y[env_idx].zero_()
        self.fleet_angle[env_idx].zero_()
        self.fleet_ships[env_idx].zero_()
        self.fleet_from_pid[env_idx].fill_(EMPTY_ID)

        self.comet_paths[env_idx].zero_()
        self.comet_path_len[env_idx].zero_()
        self.comet_group_alive[env_idx].fill_(False)
        self.comet_group_path_idx[env_idx].fill_(-1)
        self.comet_group_planet_slot[env_idx].fill_(-1)

        n = len(planets)
        if n > self.cfg.max_planets:
            raise RuntimeError(
                f"map gen produced {n} planets but max_planets={self.cfg.max_planets}"
            )
        for slot, p in enumerate(planets):
            self.planet_id[env_idx, slot] = p[0]
            self.planet_owner[env_idx, slot] = p[1]
            self.planet_x[env_idx, slot] = p[2]
            self.planet_y[env_idx, slot] = p[3]
            self.planet_radius[env_idx, slot] = p[4]
            self.planet_ships[env_idx, slot] = p[5]
            self.planet_production[env_idx, slot] = p[6]
            self.planet_alive[env_idx, slot] = True
            self.planet_initial_x[env_idx, slot] = initial_planets[slot][2]
            self.planet_initial_y[env_idx, slot] = initial_planets[slot][3]

        self._initial_planets_cpu[env_idx] = initial_planets
        self._comet_groups_cpu[env_idx] = []
        self._episode_seeds[env_idx] = seed

        self.step_count[env_idx] = 0
        self.angular_velocity[env_idx] = angular_velocity
        self.next_fleet_id[env_idx] = 0
        max_id = max((p[0] for p in planets), default=-1)
        self.next_planet_id[env_idx] = max_id + 1
        self.next_comet_group_idx[env_idx] = 0
        self.next_planet_slot[env_idx] = n
        self.done[env_idx] = False
        self.rewards[env_idx].zero_()

        self._done_cpu[env_idx] = False
        self._step_count_cpu[env_idx] = 0
        self._rewards_cpu[env_idx] = [0] * self.cfg.num_agents

    # Step pipeline --------------------------------------------------

    def step(
        self,
        actions_p0: list[list[list[float | int]]],
        actions_p1: list[list[list[float | int]]],
    ) -> None:
        """Advance one game tick across all B envs (2-player shortcut)."""

        self.step_players([actions_p0, actions_p1])

    def step_players(
        self,
        actions_per_player: list[list[list[list[float | int]]]],
    ) -> None:
        """Advance one tick. `actions_per_player[p][env_idx]` is env-p's move list."""

        if len(actions_per_player) != self.cfg.num_agents:
            raise ValueError(
                f"expected {self.cfg.num_agents} action lists, got {len(actions_per_player)}"
            )
        for p_idx, p_actions in enumerate(actions_per_player):
            if len(p_actions) != self.B:
                raise ValueError(
                    f"player {p_idx} action list has {len(p_actions)} envs, expected {self.B}"
                )

        # Sync-free short circuit: if every env is already done, no GPU work.
        if all(self._done_cpu):
            return

        self._phase_a_expire_pre_launch()
        self._phase_b_spawn_comets()
        self._phase_c_fleet_launch(actions_per_player)
        self._phase_d_production()
        planet_new_x, planet_new_y, planet_path_check = self._phase_e_planet_paths()
        combat_planet, combat_fleet = self._phase_f_fleet_movement(
            planet_new_x, planet_new_y, planet_path_check
        )
        self._phase_g_apply_planet_movement(planet_new_x, planet_new_y)
        self._phase_h_expire_mid_step()
        self._phase_i_combat(combat_planet, combat_fleet)
        self._phase_j_terminate_and_score()

        self.step_count.add_((~self.done).long())
        self._refresh_cpu_mirrors()

    # Phase A: expire comets whose path_index already exceeds length ---
    def _phase_a_expire_pre_launch(self) -> None:
        expired = self.comet_group_alive & (self.comet_group_path_idx >= self.comet_path_len)
        self._kill_comet_groups(expired)

    # Phase B: spawn new comet groups at fixed steps ------------------
    def _phase_b_spawn_comets(self) -> None:
        spawn_steps = set(COMET_SPAWN_STEPS)
        spawning: list[tuple[int, int]] = []
        for env_idx in range(self.B):
            if self._done_cpu[env_idx]:
                continue
            cur_step = self._step_count_cpu[env_idx]
            ns = cur_step + 1
            if ns not in spawn_steps:
                continue
            self._comet_groups_cpu[env_idx] = [
                g for g in self._comet_groups_cpu[env_idx] if g["expiry_step"] > cur_step
            ]
            spawning.append((env_idx, ns))
        if not spawning:
            return

        omega_cpu = self.angular_velocity.cpu().tolist()
        worker_inputs = []
        for env_idx, ns in spawning:
            live_ids: list[int] = []
            for g in self._comet_groups_cpu[env_idx]:
                live_ids.extend(g["planet_ids"])
            rng_key = f"orbit_wars-comet-{self._episode_seeds[env_idx]}-{ns}"
            worker_inputs.append(
                (
                    self._initial_planets_cpu[env_idx],
                    omega_cpu[env_idx],
                    ns,
                    live_ids,
                    self.cfg.comet_speed,
                    rng_key,
                )
            )

        results: list[tuple[list, int]]
        if len(spawning) >= self._comet_pool_threshold:
            pool = self._ensure_comet_pool()
            results = list(pool.map(_comet_worker, worker_inputs))
        else:
            results = [_comet_worker(args) for args in worker_inputs]

        for (env_idx, ns), (paths, comet_ships) in zip(spawning, results, strict=True):
            if not paths:
                continue
            self._apply_spawn(env_idx, ns, paths, comet_ships)

    def _apply_spawn(
        self,
        env_idx: int,
        spawn_step: int,
        paths: list,
        comet_ships: int,
    ) -> None:
        group_slot = int(self.next_comet_group_idx[env_idx].item())
        if group_slot >= self.G:
            return
        path_len = len(paths[0])
        if path_len > self.L:
            raise RuntimeError(f"comet path of length {path_len} exceeds max {self.L}")
        paths_cpu = torch.tensor(paths, dtype=self.dtype)
        paths_dev = paths_cpu.to(self.device, non_blocking=True)
        self.comet_paths[env_idx, group_slot, :, :path_len, :] = paths_dev
        self.comet_path_len[env_idx, group_slot] = path_len
        self.comet_group_alive[env_idx, group_slot] = True
        self.comet_group_path_idx[env_idx, group_slot] = -1

        next_pid = int(self.next_planet_id[env_idx].item())
        next_slot = int(self.next_planet_slot[env_idx].item())
        if next_slot + self.Q > self.P:
            raise RuntimeError(
                f"comet planet slot overflow at env {env_idx}: {next_slot + self.Q} > {self.P}"
            )
        slots = slice(next_slot, next_slot + self.Q)
        pids = torch.arange(next_pid, next_pid + self.Q, dtype=torch.int64, device=self.device)
        self.planet_id[env_idx, slots] = pids
        self.planet_owner[env_idx, slots] = -1
        self.planet_x[env_idx, slots] = -99.0
        self.planet_y[env_idx, slots] = -99.0
        self.planet_radius[env_idx, slots] = COMET_RADIUS
        self.planet_ships[env_idx, slots] = comet_ships
        self.planet_production[env_idx, slots] = COMET_PRODUCTION
        self.planet_alive[env_idx, slots] = True
        self.planet_is_comet[env_idx, slots] = True
        self.planet_initial_x[env_idx, slots] = -99.0
        self.planet_initial_y[env_idx, slots] = -99.0
        self.comet_group_planet_slot[env_idx, group_slot, : self.Q] = torch.arange(
            next_slot, next_slot + self.Q, dtype=torch.int64, device=self.device
        )
        slot_indices = [next_slot + q for q in range(self.Q)]
        planet_ids = [next_pid + q for q in range(self.Q)]
        for q in range(self.Q):
            self._initial_planets_cpu[env_idx].append(
                [planet_ids[q], -1, -99.0, -99.0, COMET_RADIUS, comet_ships, COMET_PRODUCTION]
            )
        self._comet_groups_cpu[env_idx].append(
            {
                "slot_indices": slot_indices,
                "planet_ids": planet_ids,
                "expiry_step": spawn_step + path_len,
            }
        )
        self.next_planet_id[env_idx] = next_pid + self.Q
        self.next_planet_slot[env_idx] = next_slot + self.Q
        self.next_comet_group_idx[env_idx] = group_slot + 1

    # Phase C: fleet launch (Python loop over moves, batched writes) ----
    def _phase_c_fleet_launch(
        self,
        actions_per_player: list[list[list[list[float | int]]]],
    ) -> None:
        if not any(any(p) for p in actions_per_player):
            return

        B = self.B
        P = self.P
        F = self.F
        num_agents = self.cfg.num_agents
        device = self.device
        ft = self.dtype

        # Hoist every per-(env, move) tensor read into a small fixed number
        # of batched CPU pulls.
        int_packed = torch.stack(
            [
                self.planet_id,
                self.planet_alive.to(torch.int64),
                self.planet_owner,
                self.planet_ships,
            ],
            dim=0,
        )
        float_packed = torch.stack(
            [self.planet_radius, self.planet_x, self.planet_y], dim=0
        )
        int_cpu = int_packed.cpu().tolist()
        float_cpu = float_packed.cpu().tolist()
        fleet_alive_cpu = self.fleet_alive.cpu().tolist()
        next_fleet_id_cpu = self.next_fleet_id.cpu().tolist()

        planet_id_cpu = int_cpu[0]
        planet_alive_cpu = int_cpu[1]
        planet_owner_cpu = int_cpu[2]
        planet_ships_cpu = int_cpu[3]
        planet_radius_cpu = float_cpu[0]
        planet_x_cpu = float_cpu[1]
        planet_y_cpu = float_cpu[2]

        free_ptr = [0] * B

        ps_envs: list[int] = []
        ps_slots: list[int] = []
        ps_new: list[int] = []
        f_envs: list[int] = []
        f_slots: list[int] = []
        f_owners: list[int] = []
        f_ids: list[int] = []
        f_xs: list[float] = []
        f_ys: list[float] = []
        f_angles: list[float] = []
        f_ships: list[int] = []
        f_from: list[int] = []

        grow_needed = False
        for env_idx in range(B):
            if self._done_cpu[env_idx]:
                continue
            env_moves = [actions_per_player[p][env_idx] for p in range(num_agents)]
            if not any(env_moves):
                continue

            ids = planet_id_cpu[env_idx]
            alive = planet_alive_cpu[env_idx]
            owners = planet_owner_cpu[env_idx]
            ships_row = planet_ships_cpu[env_idx]
            radii = planet_radius_cpu[env_idx]
            xs = planet_x_cpu[env_idx]
            ys = planet_y_cpu[env_idx]
            id_to_slot: dict[int, int] = {}
            for s in range(P):
                if alive[s]:
                    id_to_slot[ids[s]] = s

            alive_row = fleet_alive_cpu[env_idx]
            ptr = free_ptr[env_idx]
            nfid = next_fleet_id_cpu[env_idx]

            for player, moves in enumerate(env_moves):
                if not moves:
                    continue
                for move in moves:
                    if not isinstance(move, list) or len(move) != 3:
                        continue
                    from_id, angle, ships = move
                    ships_i = int(ships)
                    slot = id_to_slot.get(int(from_id))
                    if slot is None:
                        continue
                    if owners[slot] != player:
                        continue
                    cur_ships = ships_row[slot]
                    if cur_ships < ships_i or ships_i <= 0:
                        continue
                    while ptr < F and alive_row[ptr]:
                        ptr += 1
                    if ptr >= F:
                        grow_needed = True
                        break
                    fslot = ptr
                    alive_row[ptr] = True
                    ptr += 1
                    angle_f = float(angle)
                    radius = radii[slot]
                    start_x = xs[slot] + math.cos(angle_f) * (radius + 0.1)
                    start_y = ys[slot] + math.sin(angle_f) * (radius + 0.1)
                    new_ships = cur_ships - ships_i
                    ships_row[slot] = new_ships

                    ps_envs.append(env_idx)
                    ps_slots.append(slot)
                    ps_new.append(new_ships)
                    f_envs.append(env_idx)
                    f_slots.append(fslot)
                    f_owners.append(player)
                    f_ids.append(nfid)
                    f_xs.append(start_x)
                    f_ys.append(start_y)
                    f_angles.append(angle_f)
                    f_ships.append(ships_i)
                    f_from.append(int(from_id))
                    nfid += 1
                if grow_needed:
                    break

            free_ptr[env_idx] = ptr
            next_fleet_id_cpu[env_idx] = nfid
            if grow_needed:
                break

        if grow_needed:
            self._grow_fleet_buffer()
            self._phase_c_fleet_launch(actions_per_player)
            return

        if ps_envs:
            envs_t = torch.tensor(ps_envs, dtype=torch.long, device=device)
            slots_t = torch.tensor(ps_slots, dtype=torch.long, device=device)
            self.planet_ships[envs_t, slots_t] = torch.tensor(
                ps_new, dtype=torch.int64, device=device
            )
        if f_envs:
            envs_t = torch.tensor(f_envs, dtype=torch.long, device=device)
            slots_t = torch.tensor(f_slots, dtype=torch.long, device=device)
            self.fleet_owner[envs_t, slots_t] = torch.tensor(
                f_owners, dtype=torch.int64, device=device
            )
            self.fleet_id[envs_t, slots_t] = torch.tensor(
                f_ids, dtype=torch.int64, device=device
            )
            self.fleet_x[envs_t, slots_t] = torch.tensor(f_xs, dtype=ft, device=device)
            self.fleet_y[envs_t, slots_t] = torch.tensor(f_ys, dtype=ft, device=device)
            self.fleet_angle[envs_t, slots_t] = torch.tensor(
                f_angles, dtype=ft, device=device
            )
            self.fleet_ships[envs_t, slots_t] = torch.tensor(
                f_ships, dtype=torch.int64, device=device
            )
            self.fleet_from_pid[envs_t, slots_t] = torch.tensor(
                f_from, dtype=torch.int64, device=device
            )
            self.fleet_alive[envs_t, slots_t] = True
            self.next_fleet_id.copy_(
                torch.tensor(next_fleet_id_cpu, dtype=torch.int64, device=device)
            )

    def _grow_fleet_buffer(self) -> None:
        new_F = self.F * 2
        d = self.device
        ft = self.dtype
        B = self.B

        def grow(t: torch.Tensor, fill: Any) -> torch.Tensor:
            pad = torch.full((B, new_F - self.F), fill, dtype=t.dtype, device=d)
            return torch.cat([t, pad], dim=1)

        zero_ft = torch.zeros((B, new_F - self.F), dtype=ft, device=d)
        self.fleet_owner = grow(self.fleet_owner, EMPTY_OWNER)
        self.fleet_id = grow(self.fleet_id, EMPTY_ID)
        self.fleet_x = torch.cat([self.fleet_x, zero_ft], dim=1)
        self.fleet_y = torch.cat([self.fleet_y, zero_ft], dim=1)
        self.fleet_angle = torch.cat([self.fleet_angle, zero_ft], dim=1)
        self.fleet_ships = grow(self.fleet_ships, 0)
        self.fleet_from_pid = grow(self.fleet_from_pid, EMPTY_ID)
        self.fleet_alive = grow(self.fleet_alive, False)
        self.F = new_F
        logger.info("grew fleet buffer to F=%d", new_F)

    # Phase D: production tick ----------------------------------------
    def _phase_d_production(self) -> None:
        owned = self.planet_alive & (self.planet_owner >= 0)
        active = (~self.done).unsqueeze(1)
        mask = owned & active
        self.planet_ships.add_(mask.long() * self.planet_production)

    # Phase E: compute planet end-of-tick positions -------------------
    def _phase_e_planet_paths(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = self.B
        new_x = self.planet_x.clone()
        new_y = self.planet_y.clone()
        check = self.planet_alive.clone()

        dx = self.planet_initial_x - CENTER
        dy = self.planet_initial_y - CENTER
        r = torch.sqrt(dx * dx + dy * dy)
        rotating = (
            self.planet_alive
            & ~self.planet_is_comet
            & ((r + self.planet_radius) < ROTATION_RADIUS_LIMIT)
        )
        step = self.step_count.unsqueeze(1).to(self.dtype)
        omega = self.angular_velocity.unsqueeze(1)
        init_angle = torch.atan2(dy, dx)
        cur_angle = init_angle + omega * step
        rot_x = CENTER + r * torch.cos(cur_angle)
        rot_y = CENTER + r * torch.sin(cur_angle)
        new_x = torch.where(rotating, rot_x, new_x)
        new_y = torch.where(rotating, rot_y, new_y)

        self.comet_group_path_idx.add_(self.comet_group_alive.long())
        idx = self.comet_group_path_idx
        path_len = self.comet_path_len
        live = self.comet_group_alive & (idx >= 0) & (idx < path_len)
        b_arange = torch.arange(B, device=self.device).view(B, 1, 1).expand(B, self.G, self.Q)
        g_arange = (
            torch.arange(self.G, device=self.device).view(1, self.G, 1).expand(B, self.G, self.Q)
        )
        q_arange = (
            torch.arange(self.Q, device=self.device).view(1, 1, self.Q).expand(B, self.G, self.Q)
        )
        safe_idx = torch.clamp(idx, min=0, max=self.L - 1).unsqueeze(-1).expand(B, self.G, self.Q)
        comet_pos = self.comet_paths[b_arange, g_arange, q_arange, safe_idx]
        comet_new_x = comet_pos[..., 0]
        comet_new_y = comet_pos[..., 1]

        slot_idx = self.comet_group_planet_slot
        live_q = live.unsqueeze(-1).expand(B, self.G, self.Q) & (slot_idx >= 0)
        flat_slot = slot_idx.clamp(min=0)
        b_flat = b_arange
        prev_x = self.planet_x[b_flat, flat_slot]
        first_placement = prev_x < 0
        live_q_flat = live_q.reshape(B, -1)
        slot_flat = flat_slot.reshape(B, -1)
        cnx_flat = comet_new_x.reshape(B, -1)
        cny_flat = comet_new_y.reshape(B, -1)
        slot_long = slot_flat.long()
        cur_x = torch.gather(new_x, 1, slot_long)
        cur_y = torch.gather(new_y, 1, slot_long)
        cur_check = torch.gather(check, 1, slot_long)
        write_x = torch.where(live_q_flat, cnx_flat, cur_x)
        write_y = torch.where(live_q_flat, cny_flat, cur_y)
        first_placement_flat = first_placement.reshape(B, -1)
        check_write = torch.where(
            live_q_flat & first_placement_flat,
            torch.zeros_like(cur_check),
            cur_check,
        )
        new_x.scatter_(1, slot_long, write_x)
        new_y.scatter_(1, slot_long, write_y)
        check.scatter_(1, slot_long, check_write)

        return new_x, new_y, check

    # Phase F: fleet movement + swept collision -----------------------
    def _phase_f_fleet_movement(
        self,
        planet_new_x: torch.Tensor,
        planet_new_y: torch.Tensor,
        planet_check: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alive = self.fleet_alive

        ships = self.fleet_ships.clamp(min=1)
        max_speed = self.cfg.ship_speed
        log_ships = torch.log(ships.to(self.dtype))
        log1000 = math.log(1000.0)
        speed = 1.0 + (max_speed - 1.0) * (log_ships / log1000) ** 1.5
        speed = torch.clamp(speed, max=max_speed)

        old_x = self.fleet_x.clone()
        old_y = self.fleet_y.clone()
        dx = torch.cos(self.fleet_angle) * speed
        dy = torch.sin(self.fleet_angle) * speed
        new_x = old_x + dx
        new_y = old_y + dy

        Ax_old = old_x.unsqueeze(2)
        Ay_old = old_y.unsqueeze(2)
        Ax_new = new_x.unsqueeze(2)
        Ay_new = new_y.unsqueeze(2)
        Px_old = self.planet_x.unsqueeze(1)
        Py_old = self.planet_y.unsqueeze(1)
        Px_new = planet_new_x.unsqueeze(1)
        Py_new = planet_new_y.unsqueeze(1)
        Pr = self.planet_radius.unsqueeze(1)

        d0x = Ax_old - Px_old
        d0y = Ay_old - Py_old
        dvx = (Ax_new - Ax_old) - (Px_new - Px_old)
        dvy = (Ay_new - Ay_old) - (Py_new - Py_old)

        a = dvx * dvx + dvy * dvy
        b = 2.0 * (d0x * dvx + d0y * dvy)
        c = d0x * d0x + d0y * d0y - Pr * Pr

        disc = b * b - 4.0 * a * c
        small_a = a < 1e-12
        sq = torch.sqrt(disc.clamp(min=0.0))
        denom = (2.0 * a).clamp(min=1e-30)
        t1 = (-b - sq) / denom
        t2 = (-b + sq) / denom

        hit = ((disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)) | (small_a & (c <= 0.0))

        planet_valid = (self.planet_alive & planet_check).unsqueeze(1)
        fleet_valid = alive.unsqueeze(2)
        hit = hit & planet_valid & fleet_valid

        BIG = torch.full_like(t1, float("inf"))
        t1_for_argmin = torch.where(hit, t1, BIG)
        any_hit = hit.any(dim=2)
        hit_planet_idx = torch.argmin(t1_for_argmin, dim=2)

        oob = (new_x < 0.0) | (new_x > BOARD_SIZE) | (new_y < 0.0) | (new_y > BOARD_SIZE)

        seg_dx = new_x - old_x
        seg_dy = new_y - old_y
        seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy
        t_proj = torch.where(
            seg_len2 > 0.0,
            ((CENTER - old_x) * seg_dx + (CENTER - old_y) * seg_dy) / seg_len2.clamp(min=1e-30),
            torch.zeros_like(seg_len2),
        )
        t_proj = torch.clamp(t_proj, 0.0, 1.0)
        proj_x = old_x + t_proj * seg_dx
        proj_y = old_y + t_proj * seg_dy
        sun_dist = torch.sqrt((proj_x - CENTER) ** 2 + (proj_y - CENTER) ** 2)
        sun_hit = sun_dist < SUN_RADIUS

        any_hit = any_hit & alive
        oob_die = (oob | sun_hit) & alive & ~any_hit

        survives = alive & ~any_hit & ~oob_die
        self.fleet_x = torch.where(survives, new_x, old_x)
        self.fleet_y = torch.where(survives, new_y, old_y)

        self.fleet_alive = self.fleet_alive & ~any_hit & ~oob_die

        combat_planet = torch.where(any_hit, hit_planet_idx, torch.full_like(hit_planet_idx, -1))
        return combat_planet, any_hit

    # Phase G: apply planet end-of-tick positions ---------------------
    def _phase_g_apply_planet_movement(self, new_x: torch.Tensor, new_y: torch.Tensor) -> None:
        self.planet_x = new_x
        self.planet_y = new_y

    # Phase H: expire comets whose path just ended this tick ----------
    def _phase_h_expire_mid_step(self) -> None:
        expired = self.comet_group_alive & (self.comet_group_path_idx >= self.comet_path_len)
        self._kill_comet_groups(expired)

    def _kill_comet_groups(self, group_mask: torch.Tensor) -> None:
        slots = self.comet_group_planet_slot
        kill_q = group_mask.unsqueeze(-1) & (slots >= 0)
        slot_flat = slots.clamp(min=0).reshape(self.B, -1)
        kill_flat = kill_q.reshape(self.B, -1).to(torch.int64)
        kill_count = torch.zeros((self.B, self.P), dtype=torch.int64, device=self.device)
        kill_count.scatter_add_(1, slot_flat, kill_flat)
        self.planet_alive = self.planet_alive & (kill_count == 0)
        self.comet_group_alive = self.comet_group_alive & ~group_mask

    # Phase I: combat resolution --------------------------------------
    def _phase_i_combat(
        self,
        combat_planet: torch.Tensor,
        combat_fleet: torch.Tensor,
    ) -> None:
        B, P = self.B, self.P
        num_owners = self.cfg.num_agents
        arrivals = torch.zeros((B, P, num_owners), dtype=torch.int64, device=self.device)

        owner = self.fleet_owner.clamp(min=0)
        ships = self.fleet_ships
        b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand_as(combat_planet)
        planet_slot = combat_planet.clamp(min=0)
        flat = b_idx * (P * num_owners) + planet_slot * num_owners + owner
        valid = combat_fleet & (self.fleet_owner >= 0) & (self.fleet_owner < num_owners)
        flat = torch.where(valid, flat, torch.zeros_like(flat))
        ship_contrib = torch.where(valid, ships, torch.zeros_like(ships))
        arrivals.view(-1).scatter_add_(0, flat.reshape(-1), ship_contrib.reshape(-1))

        topk_vals, topk_idx = torch.topk(arrivals, k=min(2, num_owners), dim=2)
        top1_ships = topk_vals[..., 0]
        top1_owner = topk_idx[..., 0]
        if topk_vals.shape[-1] >= 2:
            top2_ships = topk_vals[..., 1]
        else:
            top2_ships = torch.zeros_like(top1_ships)

        contested = top1_ships > 0
        survivor = top1_ships - top2_ships
        tied = top1_ships == top2_ships
        survivor = torch.where(tied, torch.zeros_like(survivor), survivor)
        survivor_owner = torch.where(survivor > 0, top1_owner, torch.full_like(top1_owner, -1))

        cur_owner = self.planet_owner
        cur_ships = self.planet_ships

        same_owner = (cur_owner == survivor_owner) & contested & (survivor > 0)
        diff_owner = (cur_owner != survivor_owner) & contested & (survivor > 0)
        ships_after_friendly = cur_ships + survivor
        ships_after_hostile = cur_ships - survivor
        flips = diff_owner & (ships_after_hostile < 0)

        new_ships = cur_ships.clone()
        new_owner = cur_owner.clone()
        new_ships = torch.where(same_owner, ships_after_friendly, new_ships)
        defender_survives = diff_owner & (ships_after_hostile >= 0)
        new_ships = torch.where(defender_survives, ships_after_hostile, new_ships)
        new_ships = torch.where(flips, -ships_after_hostile, new_ships)
        new_owner = torch.where(flips, survivor_owner, new_owner)

        active_planet = self.planet_alive & contested
        self.planet_ships = torch.where(active_planet, new_ships, cur_ships)
        self.planet_owner = torch.where(active_planet, new_owner, cur_owner)

    # Phase J: termination check + scoring ----------------------------
    def _phase_j_terminate_and_score(self) -> None:
        new_step = self.step_count + (~self.done).long()
        max_steps_done = new_step >= (self.cfg.episode_steps - 2)

        owners_planet = torch.where(
            self.planet_alive,
            self.planet_owner,
            torch.full_like(self.planet_owner, EMPTY_OWNER),
        )
        owners_fleet = torch.where(
            self.fleet_alive,
            self.fleet_owner,
            torch.full_like(self.fleet_owner, EMPTY_OWNER),
        )
        num_owners = self.cfg.num_agents
        alive_per_player = torch.zeros((self.B, num_owners), dtype=torch.bool, device=self.device)
        for p in range(num_owners):
            alive_per_player[:, p] = (owners_planet == p).any(dim=1) | (owners_fleet == p).any(
                dim=1
            )
        alive_count = alive_per_player.sum(dim=1)
        one_left = alive_count <= 1

        newly_done = (~self.done) & (max_steps_done | one_left)

        scores = torch.zeros((self.B, num_owners), dtype=torch.int64, device=self.device)
        for p in range(num_owners):
            mask_p = self.planet_alive & (self.planet_owner == p)
            scores[:, p] += (mask_p.long() * self.planet_ships).sum(dim=1)
            mask_f = self.fleet_alive & (self.fleet_owner == p)
            scores[:, p] += (mask_f.long() * self.fleet_ships).sum(dim=1)

        max_score, _ = scores.max(dim=1, keepdim=True)
        is_winner = (scores == max_score) & (max_score > 0)
        rewards = torch.where(is_winner, torch.ones_like(scores), torch.full_like(scores, -1))
        self.rewards = torch.where(newly_done.unsqueeze(1), rewards, self.rewards)
        self.done = self.done | newly_done

    # Comet path generation pool -------------------------------------

    def _ensure_comet_pool(self) -> concurrent.futures.ProcessPoolExecutor:
        if self._comet_pool is None:
            max_workers = int(os.environ.get("ORBITWARS_COMET_WORKERS", "8"))
            self._comet_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            )
            logger.info("started comet path generation pool with %d workers", max_workers)
        return self._comet_pool

    def close(self) -> None:
        if self._comet_pool is not None:
            self._comet_pool.shutdown(wait=False, cancel_futures=True)
            self._comet_pool = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # CPU-mirror refresh ---------------------------------------------

    def _refresh_cpu_mirrors(self) -> None:
        """Round-trip done/step_count/rewards in a single .cpu() call."""
        num_owners = self.cfg.num_agents
        packed = torch.empty(
            (2 + num_owners, self.B), dtype=torch.int64, device=self.device
        )
        packed[0] = self.done.to(torch.int64)
        packed[1] = self.step_count
        packed[2 : 2 + num_owners] = self.rewards.t()
        rows = packed.cpu().tolist()
        self._done_cpu = [bool(v) for v in rows[0]]
        self._step_count_cpu = list(rows[1])
        rewards_rows = rows[2 : 2 + num_owners]
        for env_idx in range(self.B):
            self._rewards_cpu[env_idx] = [rewards_rows[p][env_idx] for p in range(num_owners)]

    # Observation extraction -----------------------------------------

    def dump_states(self, *, learner_player: int = 0) -> list[dict[str, Any]]:
        """Per-env state dicts compatible with `OrbitWarsEnv.dump_state`."""

        owners = self.planet_owner.cpu().tolist()
        ids = self.planet_id.cpu().tolist()
        xs = self.planet_x.cpu().tolist()
        ys = self.planet_y.cpu().tolist()
        radii = self.planet_radius.cpu().tolist()
        ships = self.planet_ships.cpu().tolist()
        prods = self.planet_production.cpu().tolist()
        alive_p = self.planet_alive.cpu().tolist()

        fl_owner = self.fleet_owner.cpu().tolist()
        fl_id = self.fleet_id.cpu().tolist()
        fl_x = self.fleet_x.cpu().tolist()
        fl_y = self.fleet_y.cpu().tolist()
        fl_angle = self.fleet_angle.cpu().tolist()
        fl_from = self.fleet_from_pid.cpu().tolist()
        fl_ships = self.fleet_ships.cpu().tolist()
        fl_alive = self.fleet_alive.cpu().tolist()
        step_cpu = self.step_count.cpu().tolist()

        out: list[dict[str, Any]] = []
        for env_idx in range(self.B):
            planets = []
            for slot in range(self.P):
                if not alive_p[env_idx][slot]:
                    continue
                planets.append(
                    [
                        float(ids[env_idx][slot]),
                        float(owners[env_idx][slot]),
                        float(xs[env_idx][slot]),
                        float(ys[env_idx][slot]),
                        float(radii[env_idx][slot]),
                        float(ships[env_idx][slot]),
                        float(prods[env_idx][slot]),
                    ]
                )
            planets.sort(key=lambda row: row[0])

            fleets = []
            for slot in range(self.F):
                if not fl_alive[env_idx][slot]:
                    continue
                fleets.append(
                    [
                        float(fl_id[env_idx][slot]),
                        float(fl_owner[env_idx][slot]),
                        float(fl_x[env_idx][slot]),
                        float(fl_y[env_idx][slot]),
                        float(fl_angle[env_idx][slot]),
                        float(fl_from[env_idx][slot]),
                        float(fl_ships[env_idx][slot]),
                    ]
                )
            fleets.sort(key=lambda row: row[0])

            out.append(
                {
                    "learner_player": int(learner_player),
                    "step": int(step_cpu[env_idx]),
                    "planets": planets,
                    "fleets": fleets,
                }
            )
        return out


__all__ = ["TorchOrbitWarsEnv", "TorchEngineConfig", "EMPTY_OWNER", "EMPTY_ID"]
