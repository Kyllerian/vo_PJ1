from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


ENV_ID = "Pendulum-v1"
SEEDS = [42, 43, 44]


@dataclass
class AlgoConfig:
    algo: str
    total_timesteps: int
    learning_rate: float
    gamma: float
    n_envs: int = 4
    policy_kwargs: dict = field(default_factory=dict)
    algo_kwargs: dict = field(default_factory=dict)
    eval_freq: int = 10_000
    eval_episodes: int = 10


def get_experiments() -> Dict[str, Dict[str, AlgoConfig]]:
    exp1 = {
        "PPO": AlgoConfig(
            algo="PPO",
            total_timesteps=200_000,
            learning_rate=3e-4,
            gamma=0.99,
            n_envs=4,
            policy_kwargs={"net_arch": [64, 64]},
            algo_kwargs={"batch_size": 64},
            eval_freq=20_000,
            eval_episodes=5,
        ),
        "SAC": AlgoConfig(
            algo="SAC",
            total_timesteps=200_000,
            learning_rate=3e-4,
            gamma=0.99,
            n_envs=1,
            policy_kwargs={"net_arch": [64, 64]},
            algo_kwargs={
                "buffer_size": 50_000,
                "batch_size": 128,
                "train_freq": 4,
                "gradient_steps": 1,
                "learning_starts": 1_000,
            },
            eval_freq=20_000,
            eval_episodes=5,
        ),
    }
    exp2 = {
        "PPO_small": AlgoConfig(
            algo="PPO",
            total_timesteps=200_000,
            learning_rate=3e-4,
            gamma=0.99,
            n_envs=4,
            policy_kwargs={"net_arch": [64, 64]},
            algo_kwargs={"batch_size": 64},
            eval_freq=20_000,
            eval_episodes=5,
        ),
        "PPO_large": AlgoConfig(
            algo="PPO",
            total_timesteps=200_000,
            learning_rate=3e-4,
            gamma=0.99,
            n_envs=4,
            policy_kwargs={"net_arch": [256, 256]},
            algo_kwargs={"batch_size": 64},
            eval_freq=20_000,
            eval_episodes=5,
        ),
    }
    return {"exp1": exp1, "exp2": exp2}
