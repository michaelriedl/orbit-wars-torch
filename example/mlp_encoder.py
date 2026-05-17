"""Reference observation encoder: one decision row per owned planet.

This is the "make-a-move-from-each-planet" encoder used in the upstream
training run. For each env, it emits one decision row per planet the
chosen seat owns. The candidate slate per row is the no-op plus the K-1
nearest non-self planets, partitioned into enemy/neutral/friendly
quotas, with launch angles pre-computed via fixed-iteration aim
prediction.

Customizing:
- Drop in your own candidate selection by editing `_class_topk` calls.
- Add features by extending `self_feat_rows` / `cand_features_rows`.
- The per-row metadata that `decode_actions` needs (`source_id`,
  `cand_id`, `cand_angle`, `cand_ships`) lives in `DecisionBatch.metadata`.
"""

# ruff: noqa: N803,N806 (uppercase tensor-shape names match the engine convention)

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from orbit_wars_torch import Move, DecisionBatch, TorchOrbitWarsEnv, ObservationEncoder
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    CENTER,
    SUN_RADIUS,
    ROTATION_RADIUS_LIMIT,
)

SELF_DIM = 11
CAND_DIM = 14
GLOBAL_DIM = 8

PLANET_LAUNCH_RADIUS_OFFSET = 0.1
AIM_MAX_ITERS = 5
MAX_FLEET_SPEED = 6.0


@dataclass(slots=True)
class MLPEncoderConfig:
    """Knobs for feature normalization + candidate slate sizing."""

    candidate_count: int = 8
    board_size: float = 100.0
    max_planets: int = 48
    max_ships: float = 400.0
    max_production: float = 5.0
    episode_steps: int = 500


@dataclass(slots=True)
class MLPPolicyInput:
    """Per-row tensor bundle consumed by `MLPPolicy.forward`."""

    self_features: torch.Tensor          # (N, SELF_DIM)
    candidate_features: torch.Tensor     # (N, K, CAND_DIM)
    global_features: torch.Tensor        # (N, GLOBAL_DIM)
    candidate_mask: torch.Tensor         # (N, K) bool


class MLPEncoder(ObservationEncoder):
    """One decision row per owned planet; K-way action over nearest planets."""

    def __init__(self, cfg: MLPEncoderConfig | None = None) -> None:
        self.cfg = cfg or MLPEncoderConfig()
        if self.cfg.candidate_count < 2:
            raise ValueError("candidate_count must be >= 2 (no-op + 1 target)")

    @torch.no_grad()
    def encode(
        self,
        engine: TorchOrbitWarsEnv,
        player_per_env: torch.Tensor,
    ) -> DecisionBatch:
        device = engine.device
        ft = engine.dtype
        B = engine.B
        P = engine.P
        K = self.cfg.candidate_count

        if player_per_env.shape != (B,):
            raise ValueError(f"player_per_env must be ({B},), got {tuple(player_per_env.shape)}")
        player_per_env = player_per_env.to(device).long()
        player_b = player_per_env.unsqueeze(1)

        planet_alive = engine.planet_alive
        planet_owner = engine.planet_owner
        planet_x = engine.planet_x
        planet_y = engine.planet_y
        planet_radius = engine.planet_radius
        planet_ships = engine.planet_ships
        planet_production = engine.planet_production
        planet_is_comet = engine.planet_is_comet
        planet_id = engine.planet_id
        fleet_alive = engine.fleet_alive
        fleet_owner = engine.fleet_owner
        fleet_ships = engine.fleet_ships
        step_count = engine.step_count
        omega = engine.angular_velocity

        my_planet_mask = planet_alive & (planet_owner == player_b)
        enemy_planet_mask = (
            planet_alive & (planet_owner >= 0) & (planet_owner != player_b)
        )
        neutral_planet_mask = planet_alive & (planet_owner == -1)

        my_planet_count = my_planet_mask.sum(dim=1)
        enemy_planet_count = enemy_planet_mask.sum(dim=1)
        neutral_planet_count = neutral_planet_mask.sum(dim=1)
        my_ships_total = (my_planet_mask.long() * planet_ships).sum(dim=1)
        enemy_ships_total = (enemy_planet_mask.long() * planet_ships).sum(dim=1)

        my_fleet_mask = fleet_alive & (fleet_owner == player_b)
        enemy_fleet_mask = fleet_alive & (fleet_owner >= 0) & (fleet_owner != player_b)
        my_fleet_ships = (my_fleet_mask.long() * fleet_ships).sum(dim=1)
        enemy_fleet_ships = (enemy_fleet_mask.long() * fleet_ships).sum(dim=1)

        env_idx, src_slot = my_planet_mask.nonzero(as_tuple=True)
        N = int(env_idx.shape[0])
        rows_per_env = my_planet_count.clone()

        if N == 0:
            return _empty_batch(B, K, device, ft, rows_per_env)

        src_x = planet_x[env_idx, src_slot]
        src_y = planet_y[env_idx, src_slot]
        src_radius = planet_radius[env_idx, src_slot]
        src_ships = planet_ships[env_idx, src_slot]
        src_production = planet_production[env_idx, src_slot]
        src_is_comet = planet_is_comet[env_idx, src_slot]
        src_id = planet_id[env_idx, src_slot]
        src_player = player_per_env[env_idx]

        src_dx_c = src_x - CENTER
        src_dy_c = src_y - CENTER
        src_orbital_r = torch.sqrt(src_dx_c * src_dx_c + src_dy_c * src_dy_c)
        src_is_rotating = ~src_is_comet & ((src_orbital_r + src_radius) < ROTATION_RADIUS_LIMIT)

        all_x = planet_x[env_idx]
        all_y = planet_y[env_idx]
        all_radius = planet_radius[env_idx]
        all_ships = planet_ships[env_idx]
        all_production = planet_production[env_idx]
        all_owner = planet_owner[env_idx]
        all_alive = planet_alive[env_idx]
        all_is_comet = planet_is_comet[env_idx]
        all_id = planet_id[env_idx]

        dx = all_x - src_x.unsqueeze(1)
        dy = all_y - src_y.unsqueeze(1)
        dist = torch.sqrt(dx * dx + dy * dy)

        slot_arange = torch.arange(P, device=device)
        self_mask = slot_arange.unsqueeze(0) == src_slot.unsqueeze(1)
        invalid_planet = self_mask | (~all_alive)
        src_p_n = src_player.unsqueeze(1)

        neutral_mask = (all_owner == -1) & ~invalid_planet
        enemy_mask = (all_owner >= 0) & (all_owner != src_p_n) & ~invalid_planet
        friendly_mask = (all_owner == src_p_n) & ~invalid_planet

        e_quota = K // 3
        n_quota = K // 3
        f_quota = K - e_quota - n_quota
        K_real = K - 1

        e_idx, e_valid = _class_topk(dist, enemy_mask, e_quota, P)
        n_idx, n_valid = _class_topk(dist, neutral_mask, n_quota, P)
        f_idx, f_valid = _class_topk(dist, friendly_mask, f_quota, P)

        cand_slot_pre = torch.cat([e_idx, n_idx, f_idx], dim=1)
        cand_valid_pre = torch.cat([e_valid, n_valid, f_valid], dim=1)
        cand_slot_pre = cand_slot_pre[:, :K_real]
        cand_valid_pre = cand_valid_pre[:, :K_real]

        zero_col = torch.zeros((N, 1), dtype=torch.long, device=device)
        cand_slot = torch.cat([zero_col, cand_slot_pre], dim=1)
        cand_present = torch.cat(
            [
                torch.ones((N, 1), dtype=torch.bool, device=device),
                cand_valid_pre,
            ],
            dim=1,
        )

        tgt_x = torch.gather(all_x, 1, cand_slot)
        tgt_y = torch.gather(all_y, 1, cand_slot)
        tgt_radius = torch.gather(all_radius, 1, cand_slot)
        tgt_ships = torch.gather(all_ships, 1, cand_slot)
        tgt_production = torch.gather(all_production, 1, cand_slot)
        tgt_owner = torch.gather(all_owner, 1, cand_slot)
        tgt_is_comet = torch.gather(all_is_comet, 1, cand_slot)
        tgt_id = torch.gather(all_id, 1, cand_slot)

        ships_needed = torch.maximum(tgt_ships + 1, torch.full_like(tgt_ships, 20))

        src_x_k = src_x.unsqueeze(1).expand_as(tgt_x)
        src_y_k = src_y.unsqueeze(1).expand_as(tgt_y)
        src_radius_k = src_radius.unsqueeze(1).expand_as(tgt_radius)

        tgt_dx_c = tgt_x - CENTER
        tgt_dy_c = tgt_y - CENTER
        tgt_orbital_r = torch.sqrt(tgt_dx_c * tgt_dx_c + tgt_dy_c * tgt_dy_c)
        tgt_is_rotating = ~tgt_is_comet & ((tgt_orbital_r + tgt_radius) < ROTATION_RADIUS_LIMIT)

        omega_n = omega[env_idx]
        omega_nk = omega_n.unsqueeze(1).expand_as(tgt_x)
        rotates = tgt_is_rotating & (omega_nk != 0.0)

        ships_for_speed = ships_needed.clamp(min=1).to(ft)
        log_ships = torch.log(ships_for_speed)
        log1000 = math.log(1000.0)
        speed_ratio = (log_ships / log1000).clamp(min=0.0, max=1.0)
        speed = 1.0 + (MAX_FLEET_SPEED - 1.0) * speed_ratio.pow(1.5)
        speed = speed.clamp(max=MAX_FLEET_SPEED).clamp(min=1e-6)

        cur_angle = torch.atan2(tgt_dy_c, tgt_dx_c)
        aim_x = tgt_x.clone()
        aim_y = tgt_y.clone()
        for _ in range(AIM_MAX_ITERS):
            center_dist = torch.sqrt(
                (aim_x - src_x_k) * (aim_x - src_x_k) + (aim_y - src_y_k) * (aim_y - src_y_k)
            )
            edge_dist = (
                center_dist - (src_radius_k + PLANET_LAUNCH_RADIUS_OFFSET) - tgt_radius
            ).clamp(min=0.0)
            turns = edge_dist / speed
            pred_angle = cur_angle + omega_nk * turns
            pred_x = CENTER + tgt_orbital_r * torch.cos(pred_angle)
            pred_y = CENTER + tgt_orbital_r * torch.sin(pred_angle)
            aim_x = torch.where(rotates, pred_x, tgt_x)
            aim_y = torch.where(rotates, pred_y, tgt_y)

        angle = torch.atan2(aim_y - src_y_k, aim_x - src_x_k)

        start_x = src_x_k + torch.cos(angle) * (src_radius_k + PLANET_LAUNCH_RADIUS_OFFSET)
        start_y = src_y_k + torch.sin(angle) * (src_radius_k + PLANET_LAUNCH_RADIUS_OFFSET)
        seg_dx = aim_x - start_x
        seg_dy = aim_y - start_y
        seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy
        seg_len2_safe = seg_len2.clamp(min=1e-12)
        t_proj = torch.where(
            seg_len2 > 0,
            ((CENTER - start_x) * seg_dx + (CENTER - start_y) * seg_dy) / seg_len2_safe,
            torch.zeros_like(seg_len2),
        )
        t_proj = t_proj.clamp(0.0, 1.0)
        proj_x = start_x + t_proj * seg_dx
        proj_y = start_y + t_proj * seg_dy
        sun_d = torch.sqrt((proj_x - CENTER) ** 2 + (proj_y - CENTER) ** 2)
        crosses_sun = sun_d < SUN_RADIUS

        src_ships_k = src_ships.unsqueeze(1).expand_as(ships_needed)
        valid_target = (
            cand_present & (ships_needed > 0) & ~crosses_sun & (src_ships_k >= ships_needed)
        )
        valid_target = valid_target.clone()
        valid_target[:, 0] = True  # no-op always selectable

        cfg = self.cfg
        board = float(cfg.board_size)
        max_ships = float(cfg.max_ships)
        max_prod = float(cfg.max_production)
        max_planets = float(cfg.max_planets)
        episode_steps = float(cfg.episode_steps)

        src_ships_norm = src_ships.to(ft).clamp(max=max_ships) / max_ships
        self_feat_rows = torch.stack(
            [
                torch.ones_like(src_x),
                src_x / board,
                src_y / board,
                src_radius / 5.0,
                src_ships_norm,
                src_production.to(ft) / max_prod,
                src_is_rotating.to(ft),
                my_planet_count[env_idx].to(ft) / max_planets,
                enemy_planet_count[env_idx].to(ft) / max_planets,
                my_ships_total[env_idx].to(ft) / (max_planets * max_ships),
                enemy_ships_total[env_idx].to(ft) / (max_planets * max_ships),
            ],
            dim=1,
        )

        tgt_ships_norm = tgt_ships.to(ft).clamp(max=max_ships) / max_ships
        tgt_is_neutral = (tgt_owner == -1).to(ft)
        tgt_is_friendly = (tgt_owner == src_p_n).to(ft)
        tgt_is_enemy = ((tgt_owner >= 0) & (tgt_owner != src_p_n)).to(ft)
        cand_dist = torch.gather(dist, 1, cand_slot)
        cand_dx = tgt_x - src_x_k
        cand_dy = tgt_y - src_y_k

        cand_features_rows = torch.stack(
            [
                torch.ones_like(tgt_x),
                tgt_is_neutral,
                tgt_is_friendly,
                tgt_is_enemy,
                tgt_x / board,
                tgt_y / board,
                cand_dx / board,
                cand_dy / board,
                cand_dist / board,
                tgt_ships_norm,
                tgt_production.to(ft) / max_prod,
                tgt_is_rotating.to(ft),
                crosses_sun.to(ft),
                src_ships_norm.unsqueeze(1).expand_as(tgt_x),
            ],
            dim=2,
        )
        present_f = cand_present.to(ft).unsqueeze(-1)
        cand_features_rows = cand_features_rows * present_f

        global_feat_per_env = torch.stack(
            [
                step_count.to(ft) / episode_steps,
                my_planet_count.to(ft) / max_planets,
                enemy_planet_count.to(ft) / max_planets,
                neutral_planet_count.to(ft) / max_planets,
                my_ships_total.to(ft) / (max_planets * max_ships),
                enemy_ships_total.to(ft) / (max_planets * max_ships),
                my_fleet_ships.to(ft) / (max_planets * max_ships),
                enemy_fleet_ships.to(ft) / (max_planets * max_ships),
            ],
            dim=1,
        )
        global_feat_rows = global_feat_per_env[env_idx]

        cand_id_safe = torch.where(cand_present, tgt_id, torch.full_like(tgt_id, -1))
        cand_ships_safe = torch.where(cand_present, ships_needed, torch.zeros_like(ships_needed))

        policy_input = MLPPolicyInput(
            self_features=self_feat_rows,
            candidate_features=cand_features_rows,
            global_features=global_feat_rows,
            candidate_mask=valid_target,
        )
        return DecisionBatch(
            policy_input=policy_input,
            env_index=env_idx,
            rows_per_env=rows_per_env,
            metadata={
                "source_id": src_id,
                "cand_id": cand_id_safe,
                "cand_angle": angle,
                "cand_ships": cand_ships_safe,
            },
        )

    @torch.no_grad()
    def decode_actions(
        self,
        batch: DecisionBatch,
        target_index: torch.Tensor,
        num_envs: int,
    ) -> list[list[Move]]:
        n = batch.num_rows
        out: list[list[Move]] = [[] for _ in range(num_envs)]
        if n == 0:
            return out
        if target_index.shape[0] != n:
            raise ValueError(
                f"target_index has {target_index.shape[0]} rows; batch has {n}"
            )

        meta = batch.metadata
        src_id = meta["source_id"]
        cand_id = meta["cand_id"]
        cand_angle = meta["cand_angle"]
        cand_ships = meta["cand_ships"]

        t_idx = target_index.to(torch.long)
        t_idx_k = t_idx.unsqueeze(1)
        cid = torch.gather(cand_id, 1, t_idx_k).squeeze(1)
        csh = torch.gather(cand_ships, 1, t_idx_k).squeeze(1)
        cang = torch.gather(cand_angle, 1, t_idx_k).squeeze(1)
        fires = (t_idx > 0) & (cid >= 0) & (csh > 0)

        # Pack four int64 fields + one float into two CPU transfers.
        int_packed = torch.stack(
            [
                batch.env_index.to(torch.int64),
                src_id.to(torch.int64),
                csh.to(torch.int64),
                fires.to(torch.int64),
            ],
            dim=0,
        )
        int_cpu = int_packed.cpu().tolist()
        angles_cpu = cang.to(torch.float32).cpu().tolist()
        env_idx_cpu = int_cpu[0]
        src_id_cpu = int_cpu[1]
        ships_cpu = int_cpu[2]
        fires_cpu = int_cpu[3]

        for r in range(n):
            if not fires_cpu[r]:
                continue
            out[env_idx_cpu[r]].append(
                [int(src_id_cpu[r]), float(angles_cpu[r]), int(ships_cpu[r])]
            )
        return out


def _class_topk(
    dist: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    num_planet_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the k nearest planets that pass `mask` per row."""

    if k <= 0:
        n = dist.shape[0]
        return (
            torch.zeros((n, 0), dtype=torch.long, device=dist.device),
            torch.zeros((n, 0), dtype=torch.bool, device=dist.device),
        )
    inf = torch.full_like(dist, float("inf"))
    masked = torch.where(mask, dist, inf)
    k_eff = min(k, num_planet_slots)
    vals, idx = torch.topk(masked, k=k_eff, dim=1, largest=False)
    valid = torch.isfinite(vals)
    return idx, valid


def _empty_batch(
    B: int,
    K: int,
    device: torch.device,
    ft: torch.dtype,
    rows_per_env: torch.Tensor,
) -> DecisionBatch:
    empty_input = MLPPolicyInput(
        self_features=torch.zeros((0, SELF_DIM), dtype=ft, device=device),
        candidate_features=torch.zeros((0, K, CAND_DIM), dtype=ft, device=device),
        global_features=torch.zeros((0, GLOBAL_DIM), dtype=ft, device=device),
        candidate_mask=torch.zeros((0, K), dtype=torch.bool, device=device),
    )
    return DecisionBatch(
        policy_input=empty_input,
        env_index=torch.zeros((0,), dtype=torch.long, device=device),
        rows_per_env=rows_per_env,
        metadata={
            "source_id": torch.zeros((0,), dtype=torch.long, device=device),
            "cand_id": torch.zeros((0, K), dtype=torch.long, device=device),
            "cand_angle": torch.zeros((0, K), dtype=ft, device=device),
            "cand_ships": torch.zeros((0, K), dtype=torch.long, device=device),
        },
    )


__all__ = [
    "MLPEncoder",
    "MLPEncoderConfig",
    "MLPPolicyInput",
    "SELF_DIM",
    "CAND_DIM",
    "GLOBAL_DIM",
]
