"""Vector-env wrapper around `TorchOrbitWarsEnv` with pluggable encoder + opponent.

`GpuVectorEnv` runs one game tick across all B envs per `step`. The
learner's action for each env comes from a sampled `target_index` over
the rows the encoder emitted; the opponent supplies actions for every
non-learner seat via the `Opponent` protocol.

This is intentionally thin -- it does not own the policy or the training
loop. Build a policy that consumes `DecisionBatch`, sample your own
`target_index` from its logits, and hand both back to `step()`.

Multi-team (`num_agents > 2`) is supported as long as `Opponent.act_seats`
returns one entry per non-learner seat.
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import torch

from .engine import TorchEngineConfig, TorchOrbitWarsEnv
from .interfaces import (
    Move,
    Opponent,
    DecisionBatch,
    ObservationEncoder,
)


@dataclass(slots=True)
class StepResult:
    """Per-env outcome of a single `step`.

    Attributes
    ----------
    reward: (B,) float. Learner reward this step (non-zero only on the
        terminating step of an episode).
    done: (B,) bool. True for envs that terminated this step (auto-reset
        already happened on the engine).
    next_batch: `DecisionBatch` encoded *after* the step (and after
        any auto-reset). Use this as the input for the next policy
        forward to keep the rollout contiguous.
    """

    reward: torch.Tensor
    done: torch.Tensor
    next_batch: DecisionBatch


class GpuVectorEnv:
    """Drive a batched torch engine with a pluggable encoder + opponent."""

    def __init__(
        self,
        engine_cfg: TorchEngineConfig,
        encoder: ObservationEncoder,
        *,
        opponent: Opponent | None = None,
        learner_seat: int = 0,
        base_seed: int = 0,
    ) -> None:
        if learner_seat < 0 or learner_seat >= engine_cfg.num_agents:
            raise ValueError(
                f"learner_seat={learner_seat} out of range for num_agents={engine_cfg.num_agents}"
            )

        self.engine = TorchOrbitWarsEnv(engine_cfg)
        self.encoder = encoder
        self.opponent = opponent
        self.learner_seat = learner_seat
        self.num_envs = engine_cfg.num_envs
        self.num_agents = engine_cfg.num_agents
        self.device = engine_cfg.device

        self._reset_seed_counters = [
            base_seed + i * 100_003 for i in range(self.num_envs)
        ]

        # Cached seat tensors; learner_seat is constant during a rollout.
        self._learner_seat_tensor = torch.full(
            (self.num_envs,), learner_seat, dtype=torch.long, device=self.device
        )
        self._opp_seat_tensors = self._build_opp_seats()
        self._learner_seat_cpu = [learner_seat] * self.num_envs
        self._opp_seat_cpu = [t.cpu().tolist() for t in self._opp_seat_tensors]

    def _build_opp_seats(self) -> list[torch.Tensor]:
        if self.num_agents <= 1:
            return []
        seats = [s for s in range(self.num_agents) if s != self.learner_seat]
        return [
            torch.full(
                (self.num_envs,), s, dtype=torch.long, device=self.device
            )
            for s in seats
        ]

    # ------------------------------------------------------------------
    # Public surface

    def reset(self, seeds: list[int] | None = None) -> DecisionBatch:
        """Reset all envs and return the initial decision batch for the learner."""

        if seeds is None:
            seeds = list(self._reset_seed_counters)
        if len(seeds) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} seeds, got {len(seeds)}")
        for i, s in enumerate(seeds):
            self._reset_seed_counters[i] = max(self._reset_seed_counters[i], int(s))
        self.engine.reset(seeds=[int(s) for s in seeds])
        return self.encoder.encode(self.engine, self._learner_seat_tensor)

    def step(
        self,
        learner_batch: DecisionBatch,
        learner_target_index: torch.Tensor,
    ) -> StepResult:
        """Apply `learner_target_index` for the learner and an opponent move, then tick.

        `learner_batch` must be the batch most recently returned by
        `reset()` or the previous `step().next_batch`. The opponent's
        action is queried fresh each step.
        """

        if learner_target_index.shape[0] != learner_batch.num_rows:
            raise ValueError(
                f"target_index has {learner_target_index.shape[0]} rows;"
                f" learner_batch has {learner_batch.num_rows}"
            )

        # --- decode learner moves ---
        learner_moves = self.encoder.decode_actions(
            learner_batch, learner_target_index, self.num_envs
        )

        # --- opponent moves for every non-learner seat ---
        if self.opponent is not None and self._opp_seat_tensors:
            opp_moves_per_seat = self.opponent.act_seats(
                self.engine, self._opp_seat_tensors
            )
        else:
            opp_moves_per_seat = [
                [[] for _ in range(self.num_envs)] for _ in self._opp_seat_tensors
            ]

        # --- assemble per-player action lists ---
        actions_per_player: list[list[list[Move]]] = [
            [[] for _ in range(self.num_envs)] for _ in range(self.num_agents)
        ]
        for env_idx in range(self.num_envs):
            actions_per_player[self.learner_seat][env_idx] = learner_moves[env_idx]
        for s, seat_cpu in enumerate(self._opp_seat_cpu):
            seat_moves = opp_moves_per_seat[s]
            for env_idx in range(self.num_envs):
                actions_per_player[seat_cpu[env_idx]][env_idx] = seat_moves[env_idx]

        # --- step the engine ---
        prev_done = self.engine.done.clone()
        self.engine.step_players(actions_per_player)
        done_now = self.engine.done
        new_done = done_now & ~prev_done

        learner_seat_unsq = self._learner_seat_tensor.unsqueeze(1)
        reward_full = (
            self.engine.rewards.to(torch.float32)
            .gather(1, learner_seat_unsq)
            .squeeze(1)
        )
        reward = torch.where(new_done, reward_full, torch.zeros_like(reward_full))

        # --- auto-reset terminal envs so the next encode sees fresh state ---
        self._auto_reset_terminal()

        next_batch = self.encoder.encode(self.engine, self._learner_seat_tensor)
        return StepResult(reward=reward, done=new_done, next_batch=next_batch)

    def close(self) -> None:
        self.engine.close()

    # ------------------------------------------------------------------
    # Internals

    def _auto_reset_terminal(self) -> None:
        for env_idx in range(self.num_envs):
            if self.engine._done_cpu[env_idx]:
                self._reset_seed_counters[env_idx] += 1
                self.engine._reset_one(env_idx, self._reset_seed_counters[env_idx])

    def update_opponent_from(self, source: Any) -> None:
        """Forward a weight sync to the opponent if it supports it."""

        if self.opponent is None:
            return
        sync = getattr(self.opponent, "sync_from", None)
        if callable(sync):
            sync(source)


__all__ = ["GpuVectorEnv", "StepResult"]
