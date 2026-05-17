"""GPU-native batched Orbit Wars environment for PyTorch.

Run B copies of the Kaggle Orbit Wars game in lockstep on a single CUDA
device. Plug in your own observation encoder and policy via the abstract
classes in `interfaces`; drive the whole thing with `GpuVectorEnv`.

Quick start:

    from orbit_wars_torch import (
        TorchEngineConfig,
        TorchOrbitWarsEnv,
        GpuVectorEnv,
    )
    from orbit_wars_torch.interfaces import (
        DecisionBatch,
        ObservationEncoder,
        Policy,
        PolicyOutput,
        Opponent,
    )

See `example/` for a reference MLP encoder + policy and a minimal PPO loop.
"""

from .engine import EMPTY_ID, EMPTY_OWNER, TorchEngineConfig, TorchOrbitWarsEnv
from .interfaces import (
    Move,
    Policy,
    Opponent,
    PolicyOutput,
    DecisionBatch,
    ObservationEncoder,
)
from .vector_env import StepResult, GpuVectorEnv

__all__ = [
    "TorchEngineConfig",
    "TorchOrbitWarsEnv",
    "EMPTY_OWNER",
    "EMPTY_ID",
    "GpuVectorEnv",
    "StepResult",
    "DecisionBatch",
    "Move",
    "ObservationEncoder",
    "Opponent",
    "Policy",
    "PolicyOutput",
]
