from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from track2.configs import SEEDS, DQNConfig, get_configs
from track2.envs.gridworld import GridWorldEnv
from utils.seed import set_global_seed


def parse_seeds(seeds_str: str) -> List[int]:
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]


def make_env(reward_mode: str, seed: int):
    def _init():
        env = GridWorldEnv(reward_mode=reward_mode)
        env.reset(seed=seed)
        return env

    return _init


def train_variant(
    variant: str,
    config: DQNConfig,
    seeds: Iterable[int],
    log_root: Path,
    model_root: Path,
    device: str,
    timesteps_override: int | None,
) -> None:
    total_timesteps = timesteps_override or config.total_timesteps
    for seed in seeds:
        set_global_seed(seed)
        run_name = f"{variant}/seed_{seed}"
        log_dir = log_root / run_name
        model_dir = model_root / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        env = DummyVecEnv([make_env(variant, seed)])
        env = VecMonitor(env, filename=str(log_dir / "monitor.csv"))

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            exploration_fraction=config.exploration_fraction,
            exploration_final_eps=config.exploration_final_eps,
            train_freq=config.train_freq,
            target_update_interval=config.target_update_interval,
            policy_kwargs=config.policy_kwargs,
            tensorboard_log=str(log_dir / "tb"),
            verbose=0,
            device=device,
        )

        model.learn(total_timesteps=total_timesteps, progress_bar=False)
        model.save(str(model_dir / "final_model"))

        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "variant": variant,
                    "seed": seed,
                    "total_timesteps": total_timesteps,
                    "learning_rate": config.learning_rate,
                    "gamma": config.gamma,
                    "buffer_size": config.buffer_size,
                    "batch_size": config.batch_size,
                    "exploration_fraction": config.exploration_fraction,
                    "exploration_final_eps": config.exploration_final_eps,
                },
                f,
                indent=2,
            )

        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="sparse,dense")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()

    configs = get_configs()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = parse_seeds(args.seeds)

    root = Path(__file__).resolve().parents[1]
    log_root = root / "track2" / "artifacts" / "logs"
    model_root = root / "track2" / "artifacts" / "models"

    for variant in variants:
        train_variant(
            variant,
            configs[variant],
            seeds,
            log_root,
            model_root,
            args.device,
            args.timesteps,
        )


if __name__ == "__main__":
    main()
