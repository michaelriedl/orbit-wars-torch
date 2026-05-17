# orbit-wars-torch

A GPU-native, batched PyTorch port of the Kaggle [Orbit Wars](https://www.kaggle.com/competitions) game engine. Runs `B` games in lockstep as a single tensor program so you can drive hundreds of envs at thousands of steps per second on a single GPU.

The engine is fully reusable: bring your own observation encoder and policy via the `ObservationEncoder` / `Policy` / `Opponent` interfaces and you have a ready-to-use RL environment.

## What's in here

```
kaggle_share/
├── orbit_wars_torch/         # The reusable env package
│   ├── engine.py             # TorchOrbitWarsEnv — batched GPU game engine
│   ├── interfaces.py         # ObservationEncoder, Policy, Opponent ABCs
│   └── vector_env.py         # GpuVectorEnv — drives engine + encoder + opponent
└── example/                  # Reference implementation
    ├── mlp_encoder.py        # Per-planet decision encoder (concrete ObservationEncoder)
    ├── mlp_policy.py         # Small MLP policy (concrete Policy)
    ├── opponents.py          # NoOpOpponent + SelfPlayOpponent
    └── train_ppo.py          # End-to-end PPO loop (~250 lines)
```

## Requirements

- Python 3.10+
- `torch` (CUDA recommended; CPU works for small batches)
- `kaggle-environments` (the game rules)

```bash
pip install torch kaggle-environments
```

## Quickstart

Train with the reference setup:

```bash
python -m example.train_ppo --num-envs 64 --updates 50
```

You should see throughput climb to multiple thousand env-steps/sec on a single GPU. The reference loop runs self-play with a deterministic opponent that syncs from the learner every 5 updates.

## Bring-your-own encoder and policy

Three abstract classes you implement:

```python
from orbit_wars_torch import (
    DecisionBatch,        # what your encoder produces
    ObservationEncoder,   # turns engine state -> DecisionBatch
    Policy,               # nn.Module that consumes a DecisionBatch
    PolicyOutput,         # (target_logits, value)
    Opponent,             # chooses moves for non-learner seats
)
```

### `ObservationEncoder`

```python
class MyEncoder(ObservationEncoder):
    def encode(self, engine, player_per_env) -> DecisionBatch:
        # Read engine.planet_*, engine.fleet_*, etc. — all (B, ...) tensors.
        # Build whatever your policy consumes; stash it in DecisionBatch.policy_input.
        # Stash per-row metadata needed to decode actions in DecisionBatch.metadata.
        ...

    def decode_actions(self, batch, target_index, num_envs) -> list[list[Move]]:
        # Map sampled action indices back to engine moves
        # ([from_planet_id, angle, ships]).
        ...
```

### `Policy`

```python
class MyPolicy(Policy):
    def forward(self, batch: DecisionBatch) -> PolicyOutput:
        x = batch.policy_input
        # ... your network here ...
        return PolicyOutput(target_logits=logits, value=value)
```

### Drive it

```python
from orbit_wars_torch import GpuVectorEnv, TorchEngineConfig

engine_cfg = TorchEngineConfig(num_envs=128, device=torch.device("cuda"))
env = GpuVectorEnv(engine_cfg, encoder=MyEncoder(), opponent=MyOpponent())

batch = env.reset()
for _ in range(rollout_steps):
    out = my_policy(batch)
    target_idx = sample_from_logits(out.target_logits)
    step = env.step(batch, target_idx)
    # step.reward, step.done -> (B,) tensors
    # step.next_batch       -> already encoded for the next forward
    batch = step.next_batch
```

## Engine notes

- **State layout.** All game state lives on `engine.device` as `(B, ...)` tensors. Planet buffer is fixed size; fleet buffer auto-grows in powers of two.
- **Per-step pipeline.** Comet expiry/spawn → fleet launch → production → planet movement → swept-pair fleet/planet collision → combat resolution → termination + scoring. Each phase is a single batched kernel.
- **CPU touches.** Map generation, comet path generation, and the fleet-launch dispatch loop run on CPU because they're irregular work; everything per-tick stays on the GPU.
- **Multi-team.** Set `TorchEngineConfig.num_agents=4` for 4-team Orbit Wars. Your `Opponent.act_seats` will be asked for one move list per non-learner seat.

## Customizing the example

- **Encoder.** Edit `example/mlp_encoder.py`: change which candidates the policy sees (the `_class_topk` calls), add features (extend `self_feat_rows` and `cand_features_rows`), or change normalization knobs in `MLPEncoderConfig`.
- **Policy.** Drop in any `nn.Module` that accepts a `DecisionBatch` and returns a `PolicyOutput`. The reference is a 3-tower MLP; a transformer would slot in just as easily.
- **Opponent.** Subclass `Opponent`. If you can batch multiple seats in one forward (like `SelfPlayOpponent`), override `act_seats`; otherwise the default falls back to per-seat `act` calls.

## File-level pointers

- Step pipeline: `orbit_wars_torch/engine.py` `_phase_a_expire_pre_launch` through `_phase_j_terminate_and_score`
- Auto-reset on episode end: `orbit_wars_torch/vector_env.py:_auto_reset_terminal`
- Concrete encoder reference: `example/mlp_encoder.py:MLPEncoder.encode`
- Concrete training loop: `example/train_ppo.py:main`

## License

MIT (drop in whichever LICENSE you ship with).
