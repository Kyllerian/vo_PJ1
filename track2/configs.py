from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


SEEDS = [42, 43, 44]


@dataclass
class DQNConfig:
    total_timesteps: int
    learning_rate: float
    gamma: float
    buffer_size: int
    batch_size: int
    exploration_fraction: float
    exploration_final_eps: float
    train_freq: int = 1
    target_update_interval: int = 500
    policy_kwargs: dict = field(default_factory=dict)


def get_configs() -> Dict[str, DQNConfig]:
    base = DQNConfig(
        total_timesteps=100_000,
        learning_rate=1e-3,
        gamma=0.99,
        buffer_size=50_000,
        batch_size=64,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        train_freq=1,
        target_update_interval=500,
        policy_kwargs={"net_arch": [64, 64]},
    )
    return {"sparse": base, "dense": base}

