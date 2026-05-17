"""Reference opponents: no-op (idle) and co-batched self-play.

`NoOpOpponent` issues empty move lists -- handy for measuring engine
throughput in isolation, or for a curriculum starting point.

`SelfPlayOpponent` runs the same policy used by the learner across all
non-learner seats with a single fused forward pass. It owns its own copy
of the policy weights; call `sync_from` at the cadence your training
loop prefers (e.g., every K updates) to mirror the learner.
"""

from __future__ import annotations

import torch
from orbit_wars_torch import (
    Move,
    Policy,
    Opponent,
    PolicyOutput,
    DecisionBatch,
    TorchOrbitWarsEnv,
    ObservationEncoder,
)
from torch.distributions import Categorical


class NoOpOpponent(Opponent):
    """Always passes (empty move list per env)."""

    def act(
        self,
        engine: TorchOrbitWarsEnv,
        seat_per_env: torch.Tensor,
    ) -> list[list[Move]]:
        return [[] for _ in range(engine.B)]


class SelfPlayOpponent(Opponent):
    """Run the same encoder/policy as the learner on every opponent seat.

    Multi-seat games (4-team) get one fused forward across all (env, seat)
    rows. The opponent owns its own policy copy; weights stay in sync with
    the learner via `sync_from`.
    """

    def __init__(
        self,
        encoder: ObservationEncoder,
        policy: Policy,
        device: torch.device,
        deterministic: bool = True,
    ) -> None:
        self.encoder = encoder
        self.policy = policy.to(device)
        self.policy.eval()
        self.device = device
        self.deterministic = deterministic

    def sync_from(self, source: Policy) -> None:
        self.policy.load_state_dict(source.state_dict())
        self.policy.eval()

    @torch.no_grad()
    def act(
        self,
        engine: TorchOrbitWarsEnv,
        seat_per_env: torch.Tensor,
    ) -> list[list[Move]]:
        batch = self.encoder.encode(engine, seat_per_env)
        if batch.num_rows == 0:
            return [[] for _ in range(engine.B)]
        out = self.policy(batch)
        target_idx = _sample(out, self.deterministic)
        return self.encoder.decode_actions(batch, target_idx, engine.B)

    @torch.no_grad()
    def act_seats(
        self,
        engine: TorchOrbitWarsEnv,
        seats_per_env: list[torch.Tensor],
    ) -> list[list[list[Move]]]:
        if not seats_per_env:
            return []
        parts = [self.encoder.encode(engine, seat) for seat in seats_per_env]
        n_per = [p.num_rows for p in parts]
        total = int(sum(n_per))
        if total == 0:
            return [[[] for _ in range(engine.B)] for _ in seats_per_env]
        merged = _concat_batches(parts)
        out = self.policy(merged)
        target_idx_all = _sample(out, self.deterministic)

        # Decode per seat by slicing the concatenated target tensor back
        # into per-part chunks. Each part owns its own metadata.
        result: list[list[list[Move]]] = []
        offset = 0
        for part, n in zip(parts, n_per, strict=True):
            if n == 0:
                result.append([[] for _ in range(engine.B)])
                continue
            seg = target_idx_all[offset : offset + n]
            result.append(self.encoder.decode_actions(part, seg, engine.B))
            offset += n
        return result


def _sample(outputs: PolicyOutput, deterministic: bool) -> torch.Tensor:
    logits = outputs.target_logits
    invalid = ~torch.isfinite(logits).any(dim=-1)
    if invalid.any():
        logits = logits.clone()
        logits[invalid] = 0.0
    if deterministic:
        return logits.argmax(dim=-1)
    return Categorical(logits=logits).sample()


def _concat_batches(parts: list[DecisionBatch]) -> DecisionBatch:
    """Row-concat several batches that share encoder schema.

    The metadata dict is concatenated key-by-key along dim 0. This
    keeps `decode_actions` for each part working on its slice without
    the caller needing to know the schema.
    """

    from .mlp_encoder import MLPPolicyInput

    inputs = [p.policy_input for p in parts]
    merged_input = MLPPolicyInput(
        self_features=torch.cat([i.self_features for i in inputs], dim=0),
        candidate_features=torch.cat([i.candidate_features for i in inputs], dim=0),
        global_features=torch.cat([i.global_features for i in inputs], dim=0),
        candidate_mask=torch.cat([i.candidate_mask for i in inputs], dim=0),
    )
    rows_per_env = parts[0].rows_per_env.clone()
    for p in parts[1:]:
        rows_per_env = rows_per_env + p.rows_per_env
    meta_keys = list(parts[0].metadata.keys())
    merged_meta = {
        k: torch.cat([p.metadata[k] for p in parts], dim=0) for k in meta_keys
    }
    return DecisionBatch(
        policy_input=merged_input,
        env_index=torch.cat([p.env_index for p in parts], dim=0),
        rows_per_env=rows_per_env,
        metadata=merged_meta,
    )


__all__ = ["NoOpOpponent", "SelfPlayOpponent"]
