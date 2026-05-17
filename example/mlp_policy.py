"""Reference policy: per-planet K-way target head + scalar value head.

Subclass of `orbit_wars_torch.Policy` so it slots into the env wrapper
out of the box. The forward consumes the `MLPPolicyInput` produced by
`MLPEncoder`.
"""

from __future__ import annotations

import torch
from torch import nn
from orbit_wars_torch import Policy, PolicyOutput, DecisionBatch

from .mlp_encoder import CAND_DIM, SELF_DIM, GLOBAL_DIM, MLPPolicyInput


class MLPPolicy(Policy):
    """Per-planet PPO policy with a K-way target head and a scalar value head."""

    def __init__(
        self,
        candidate_count: int,
        hidden_size: int = 128,
        self_dim: int = SELF_DIM,
        candidate_dim: int = CAND_DIM,
        global_dim: int = GLOBAL_DIM,
    ) -> None:
        super().__init__()
        self.candidate_count = candidate_count
        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.target_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, batch: DecisionBatch) -> PolicyOutput:  # type: ignore[override]
        inp: MLPPolicyInput = batch.policy_input
        self_hidden = self.self_encoder(inp.self_features)
        global_hidden = self.global_encoder(inp.global_features)
        candidate_hidden = self.candidate_encoder(inp.candidate_features)
        expanded_self = self_hidden.unsqueeze(1).expand(-1, self.candidate_count, -1)
        expanded_global = global_hidden.unsqueeze(1).expand(-1, self.candidate_count, -1)
        joint = torch.cat([expanded_self, expanded_global, candidate_hidden], dim=-1)
        target_logits = self.target_head(joint).squeeze(-1)
        target_logits = target_logits.masked_fill(
            ~inp.candidate_mask, torch.finfo(target_logits.dtype).min
        )
        pooled_candidates = candidate_hidden.mean(dim=1)
        value = self.value_head(
            torch.cat([self_hidden, global_hidden, pooled_candidates], dim=-1)
        ).squeeze(-1)
        return PolicyOutput(target_logits=target_logits, value=value)


__all__ = ["MLPPolicy"]
