"""Abstract interfaces for plugging in your own encoder, policy, and opponent.

The `GpuVectorEnv` wrapper drives `TorchOrbitWarsEnv` against any
combination of an `ObservationEncoder` + `Policy` + `Opponent`. Each
abstract type below documents the contract you need to satisfy.

Conventions:

- `engine` is a `TorchOrbitWarsEnv`. All its state lives on
  `engine.device`; reads should stay on-device when possible.
- `player_per_env` / `seat_per_env` is a `(B,)` long tensor on the engine
  device naming which seat (0..num_agents-1) the call refers to in each
  env.
- A `Move` is a 3-element list `[from_planet_id, angle, ships]` -- the
  same shape the Kaggle env consumes.
- The engine's per-step entry point takes `actions_per_player[seat][env]`
  as a list of `Move` lists.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

import torch
from torch import nn

Move = list[float | int]


@dataclass(slots=True)
class PolicyOutput:
    """Standard policy head output.

    Attributes
    ----------
    target_logits: (N, A) action logits, one row per decision point.
        Invalid actions should already be masked with `-inf` here so the
        sampler ignores them.
    value: (N,) scalar value estimate per row.
    """

    target_logits: torch.Tensor
    value: torch.Tensor


@dataclass(slots=True)
class DecisionBatch:
    """One row per (env, decision-point) plus encoder-defined payload.

    The env wrapper only reads `env_index` and `rows_per_env`; everything
    else is opaque. Stash whatever your `Policy.forward` needs in
    `policy_input` and whatever your decoder needs to turn a sampled
    action back into a `Move` in `metadata`.

    Attributes
    ----------
    policy_input: Encoder-specific structure passed to `Policy.forward`.
    env_index: (N,) long. Which env each row belongs to.
    rows_per_env: (B,) long. Row count per env, summing to N.
    metadata: Free-form dict for the encoder's own use during decode.
    """

    policy_input: Any
    env_index: torch.Tensor
    rows_per_env: torch.Tensor
    metadata: dict[str, Any]

    @property
    def num_rows(self) -> int:
        return int(self.env_index.shape[0])


class ObservationEncoder(ABC):
    """Turn engine state into a `DecisionBatch` and decode actions back.

    Implementations decide what counts as a "decision point" (one per
    owned planet? one per env? something else) and what tensors land in
    `policy_input`. Both `encode` and `decode_actions` must be
    tensor-native; per-env Python loops defeat the purpose of the GPU
    engine.
    """

    @abstractmethod
    def encode(
        self,
        engine: Any,
        player_per_env: torch.Tensor,
    ) -> DecisionBatch:
        """Build per-row inputs for the seat each env names.

        Envs whose seat has no decision points contribute zero rows. The
        result must satisfy `int(rows_per_env.sum()) == num_rows`.
        """

    @abstractmethod
    def decode_actions(
        self,
        batch: DecisionBatch,
        target_index: torch.Tensor,
        num_envs: int,
    ) -> list[list[Move]]:
        """Map a sampled action index per row back to per-env move lists.

        `target_index` is `(N,)` long. The returned list has length
        `num_envs`; each inner list holds zero or more `Move`s.
        """


class Policy(nn.Module, ABC):
    """A neural network that consumes a `DecisionBatch.policy_input`.

    Subclasses must override `forward(batch)` -- not `forward(*args)`. The
    extra `batch` indirection (over taking unpacked tensors) lets the env
    wrapper hand the same object to encoder, policy, and decoder without
    knowing the encoder's schema.
    """

    @abstractmethod
    def forward(self, batch: DecisionBatch) -> PolicyOutput:  # type: ignore[override]
        ...


class Opponent(ABC):
    """Picks actions for one or more non-learner seats each step.

    `seat_per_env` names the opponent's seat in each env. For multi-seat
    opponents (e.g., a 4-team game where the opponent owns three seats),
    call the opponent once per seat or implement `act_seats` for a fused
    forward.
    """

    @abstractmethod
    def act(
        self,
        engine: Any,
        seat_per_env: torch.Tensor,
    ) -> list[list[Move]]:
        """Return per-env move lists for the named seat."""

    def act_seats(
        self,
        engine: Any,
        seats_per_env: list[torch.Tensor],
    ) -> list[list[list[Move]]]:
        """Multi-seat default: just calls `act` per seat.

        Override this if you can batch multiple seats into a single
        policy forward (see `example.SelfPlayOpponent`).
        """

        return [self.act(engine, seat) for seat in seats_per_env]


__all__ = [
    "DecisionBatch",
    "Move",
    "ObservationEncoder",
    "Opponent",
    "Policy",
    "PolicyOutput",
]
