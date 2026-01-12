from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from track1.configs import ENV_ID, SEEDS, AlgoConfig, get_experiments
from utils.seed import set_global_seed


ALGOS = {
    "PPO": PPO,
    "SAC": SAC,
}


def parse_seeds(seeds_str: str) -> List[int]:
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]


def train_variant(
    experiment: str,
    variant: str,
    config: AlgoConfig,
    seeds: Iterable[int],
    log_root: Path,
    model_root: Path,
    device: str,
    timesteps_override: int | None,
) -> None:
    total_timesteps = timesteps_override or config.total_timesteps
    for seed in seeds:
        set_global_seed(seed)
        run_name = f"{experiment}/{variant}/seed_{seed}"
        log_dir = log_root / run_name
        model_dir = model_root / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        def make_env(base_seed: int, rank: int):
            def _init():
                env = gym.make(ENV_ID)
                env.reset(seed=base_seed + rank)
                return env

            return _init

        env_fns = [make_env(seed, i) for i in range(config.n_envs)]
        env = DummyVecEnv(env_fns)
        env = VecMonitor(env, filename=str(log_dir / "monitor.csv"))
        eval_env = DummyVecEnv([make_env(seed + 1000, 0)])
        eval_env = VecMonitor(eval_env, filename=str(log_dir / "eval_monitor.csv"))

        algo_cls = ALGOS[config.algo]
        model = algo_cls(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            policy_kwargs=config.policy_kwargs,
            tensorboard_log=str(log_dir / "tb"),
            verbose=0,
            device=device,
            **config.algo_kwargs,
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(log_dir / "eval"),
            eval_freq=config.eval_freq,
            n_eval_episodes=config.eval_episodes,
            deterministic=True,
        )

        model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=False)
        model.save(str(model_dir / "final_model"))

        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment": experiment,
                    "variant": variant,
                    "seed": seed,
                    "env_id": ENV_ID,
                    "total_timesteps": total_timesteps,
                    "learning_rate": config.learning_rate,
                    "gamma": config.gamma,
                    "n_envs": config.n_envs,
                    "policy_kwargs": config.policy_kwargs,
                    "algo_kwargs": config.algo_kwargs,
                },
                f,
                indent=2,
            )

        env.close()
        eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["exp1", "exp2"], default="exp1")
    parser.add_argument("--variants", default=None)
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()

    experiments = get_experiments()
    configs = experiments[args.experiment]
    seeds = parse_seeds(args.seeds)

    root = Path(__file__).resolve().parents[1]
    log_root = root / "track1" / "artifacts" / "logs"
    model_root = root / "track1" / "artifacts" / "models"

    if args.variants:
        selected = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        selected = list(configs.keys())

    for variant in selected:
        config = configs[variant]
        train_variant(
            args.experiment,
            variant,
            config,
            seeds,
            log_root,
            model_root,
            args.device,
            args.timesteps,
        )


if __name__ == "__main__":
    main()
