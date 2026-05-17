"""Minimal PPO training loop for orbit_wars_torch.

End-to-end: spin up a `GpuVectorEnv`, run rollouts with the MLP encoder
and policy, do a few PPO updates, repeat. Roughly 200 lines of training
code on top of the env package.

Run:
    python -m example.train_ppo --num-envs 64 --updates 50

Defaults to CUDA when available. Self-play is on by default; the
opponent's policy weights track the learner with a configurable
interval.
"""

from __future__ import annotations

import time
import argparse
from dataclasses import dataclass

import torch
from torch import nn
from orbit_wars_torch import GpuVectorEnv, DecisionBatch, TorchEngineConfig
from torch.distributions import Categorical

from .opponents import NoOpOpponent, SelfPlayOpponent
from .mlp_policy import MLPPolicy
from .mlp_encoder import MLPEncoder, MLPPolicyInput, MLPEncoderConfig


@dataclass(slots=True)
class PPOConfig:
    rollout_steps: int = 32
    num_envs: int = 64
    total_updates: int = 50
    epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    candidate_count: int = 8
    hidden_size: int = 128
    opponent: str = "self"   # "self" | "noop"
    self_play_update_interval: int = 5
    seed: int = 42
    log_every: int = 1


@dataclass(slots=True)
class _StepRecord:
    """Per-step rollout data, kept on device until the rollout finishes."""

    policy_input: MLPPolicyInput
    target_index: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    env_index: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


def main() -> None:
    cfg = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    encoder = MLPEncoder(MLPEncoderConfig(candidate_count=cfg.candidate_count))
    learner = MLPPolicy(candidate_count=cfg.candidate_count, hidden_size=cfg.hidden_size).to(device)
    optimizer = torch.optim.Adam(learner.parameters(), lr=cfg.lr)

    opponent: NoOpOpponent | SelfPlayOpponent
    if cfg.opponent == "noop":
        opponent = NoOpOpponent()
    elif cfg.opponent == "self":
        opp_policy = MLPPolicy(
            candidate_count=cfg.candidate_count, hidden_size=cfg.hidden_size
        )
        opp_policy.load_state_dict(learner.state_dict())
        opponent = SelfPlayOpponent(encoder=encoder, policy=opp_policy, device=device)
    else:
        raise ValueError(f"unknown opponent: {cfg.opponent}")

    engine_cfg = TorchEngineConfig(num_envs=cfg.num_envs, device=device)
    env = GpuVectorEnv(
        engine_cfg, encoder, opponent=opponent, learner_seat=0, base_seed=cfg.seed
    )

    batch = env.reset()
    t0 = time.time()
    for update in range(1, cfg.total_updates + 1):
        records, batch, stats = _collect_rollout(env, learner, batch, cfg, device)
        loss_stats = _ppo_update(learner, optimizer, records, batch, cfg, device)
        if isinstance(opponent, SelfPlayOpponent) and update % cfg.self_play_update_interval == 0:
            opponent.sync_from(learner)

        if update % cfg.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"update={update:04d} "
                f"reward_mean={stats['reward_mean']:+.3f} "
                f"episodes={int(stats['episodes'])} "
                f"samples={int(stats['samples'])} "
                f"loss={loss_stats['loss']:.3f} "
                f"pg={loss_stats['pg']:.3f} "
                f"vf={loss_stats['vf']:.3f} "
                f"ent={loss_stats['ent']:.3f} "
                f"steps/s={cfg.num_envs * cfg.rollout_steps * update / max(elapsed, 1e-6):.0f}"
            )

    env.close()


# ----------------------------------------------------------------------
# Rollout


def _collect_rollout(
    env: GpuVectorEnv,
    learner: MLPPolicy,
    initial_batch: DecisionBatch,
    cfg: PPOConfig,
    device: torch.device,
) -> tuple[list[_StepRecord], DecisionBatch, dict[str, float]]:
    """Run `cfg.rollout_steps` ticks. Returns step records, the last batch, and stats."""

    records: list[_StepRecord] = []
    running_reward = torch.zeros((env.num_envs,), device=device)
    ep_reward_sum = torch.zeros((), device=device)
    ep_count = torch.zeros((), device=device, dtype=torch.int64)

    batch = initial_batch
    for _ in range(cfg.rollout_steps):
        if batch.num_rows > 0:
            with torch.no_grad():
                out = learner(batch)
                target_idx, log_prob, value = _sample(out)
        else:
            empty = torch.zeros((0,), dtype=torch.long, device=device)
            target_idx = empty
            log_prob = torch.zeros((0,), device=device)
            value = torch.zeros((0,), device=device)

        step = env.step(batch, target_idx)
        records.append(
            _StepRecord(
                policy_input=batch.policy_input,
                target_index=target_idx,
                log_prob=log_prob,
                value=value,
                env_index=batch.env_index,
                reward=step.reward,
                done=step.done,
            )
        )

        running_reward = running_reward + step.reward
        done_f = step.done.to(torch.float32)
        ep_reward_sum = ep_reward_sum + (running_reward * done_f).sum()
        ep_count = ep_count + step.done.to(torch.int64).sum()
        running_reward = running_reward * (1.0 - done_f)

        batch = step.next_batch

    ep_n = int(ep_count.item())
    reward_mean = float((ep_reward_sum / ep_count.clamp(min=1)).item()) if ep_n > 0 else 0.0
    total_samples = sum(r.target_index.shape[0] for r in records)
    return records, batch, {
        "reward_mean": reward_mean,
        "episodes": float(ep_n),
        "samples": float(total_samples),
    }


def _sample(out) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = out.target_logits
    invalid = ~torch.isfinite(logits).any(dim=-1)
    if invalid.any():
        logits = logits.clone()
        logits[invalid] = 0.0
    dist = Categorical(logits=logits)
    idx = dist.sample()
    return idx, dist.log_prob(idx), out.value


# ----------------------------------------------------------------------
# PPO update


def _ppo_update(
    learner: MLPPolicy,
    optimizer: torch.optim.Optimizer,
    records: list[_StepRecord],
    final_batch: DecisionBatch,
    cfg: PPOConfig,
    device: torch.device,
) -> dict[str, float]:
    """Compute per-env discounted returns + GAE-style advantages, then PPO clip update."""

    # Bootstrap value at end of rollout (mean across decision rows per env).
    num_envs = records[0].reward.shape[0]
    if final_batch.num_rows > 0:
        with torch.no_grad():
            boot = learner(final_batch)
        per_env_value = torch.zeros((num_envs,), device=device)
        counts = torch.zeros((num_envs,), device=device)
        per_env_value.scatter_add_(0, final_batch.env_index, boot.value)
        counts.scatter_add_(
            0, final_batch.env_index, torch.ones_like(boot.value)
        )
        per_env_value = torch.where(
            counts > 0, per_env_value / counts.clamp(min=1), torch.zeros_like(per_env_value)
        )
    else:
        per_env_value = torch.zeros((num_envs,), device=device)

    # Walk reverse, propagate discounted return, slice per-row.
    future_return = per_env_value.clone()
    rows_target_idx: list[torch.Tensor] = []
    rows_log_prob: list[torch.Tensor] = []
    rows_value: list[torch.Tensor] = []
    rows_return: list[torch.Tensor] = []
    rows_advantage: list[torch.Tensor] = []
    rows_input_self: list[torch.Tensor] = []
    rows_input_cand: list[torch.Tensor] = []
    rows_input_global: list[torch.Tensor] = []
    rows_input_mask: list[torch.Tensor] = []
    for rec in reversed(records):
        future_return = (
            rec.reward + cfg.gamma * future_return * (1.0 - rec.done.float())
        )
        if rec.target_index.shape[0] == 0:
            continue
        per_row_return = future_return[rec.env_index]
        per_row_adv = per_row_return - rec.value
        rows_target_idx.append(rec.target_index)
        rows_log_prob.append(rec.log_prob)
        rows_value.append(rec.value)
        rows_return.append(per_row_return)
        rows_advantage.append(per_row_adv)
        rows_input_self.append(rec.policy_input.self_features)
        rows_input_cand.append(rec.policy_input.candidate_features)
        rows_input_global.append(rec.policy_input.global_features)
        rows_input_mask.append(rec.policy_input.candidate_mask)

    if not rows_target_idx:
        return {"loss": 0.0, "pg": 0.0, "vf": 0.0, "ent": 0.0}

    target_idx = torch.cat(rows_target_idx, dim=0)
    old_log_prob = torch.cat(rows_log_prob, dim=0)
    returns = torch.cat(rows_return, dim=0)
    advantages = torch.cat(rows_advantage, dim=0)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    self_features = torch.cat(rows_input_self, dim=0)
    candidate_features = torch.cat(rows_input_cand, dim=0)
    global_features = torch.cat(rows_input_global, dim=0)
    candidate_mask = torch.cat(rows_input_mask, dim=0)

    n_rows = target_idx.shape[0]
    last = {"loss": 0.0, "pg": 0.0, "vf": 0.0, "ent": 0.0}
    for _ in range(cfg.epochs):
        perm = torch.randperm(n_rows, device=device)
        for start in range(0, n_rows, cfg.minibatch_size):
            idx = perm[start : start + cfg.minibatch_size]
            mb_batch = DecisionBatch(
                policy_input=MLPPolicyInput(
                    self_features=self_features[idx],
                    candidate_features=candidate_features[idx],
                    global_features=global_features[idx],
                    candidate_mask=candidate_mask[idx],
                ),
                env_index=torch.zeros((idx.shape[0],), dtype=torch.long, device=device),
                rows_per_env=torch.zeros((0,), dtype=torch.long, device=device),
                metadata={},
            )
            out = learner(mb_batch)
            dist = Categorical(logits=out.target_logits)
            new_log_prob = dist.log_prob(target_idx[idx])
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_prob - old_log_prob[idx])
            adv_mb = advantages[idx]
            pg = -torch.min(
                ratio * adv_mb,
                torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * adv_mb,
            ).mean()
            vf = ((out.value - returns[idx]) ** 2).mean()
            loss = pg + cfg.vf_coef * vf - cfg.ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(learner.parameters(), cfg.max_grad_norm)
            optimizer.step()
            last = {
                "loss": float(loss.item()),
                "pg": float(pg.item()),
                "vf": float(vf.item()),
                "ent": float(entropy.item()),
            }
    return last


def _parse_args() -> PPOConfig:
    p = argparse.ArgumentParser(description="Minimal PPO loop for orbit_wars_torch")
    p.add_argument("--rollout-steps", type=int, default=32)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--updates", type=int, default=50, dest="total_updates")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--opponent", choices=["self", "noop"], default="self")
    p.add_argument("--self-play-update-interval", type=int, default=5)
    p.add_argument("--candidate-count", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return PPOConfig(
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        total_updates=args.total_updates,
        lr=args.lr,
        minibatch_size=args.minibatch_size,
        epochs=args.epochs,
        opponent=args.opponent,
        self_play_update_interval=args.self_play_update_interval,
        candidate_count=args.candidate_count,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
