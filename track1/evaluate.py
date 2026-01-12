from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from track1.configs import ENV_ID, SEEDS, get_experiments
from utils.seed import set_global_seed
from utils.video import record_episode


ALGOS = {
    "PPO": PPO,
    "SAC": SAC,
}


def parse_seeds(seeds_str: str) -> List[int]:
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]


def evaluate_variant(
    experiment: str,
    variant: str,
    algo_name: str,
    seeds: Iterable[int],
    model_root: Path,
    n_eval_episodes: int,
    model_type: str,
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        model_dir = model_root / experiment / variant / f"seed_{seed}"
        model_path = model_dir / f"{model_type}_model.zip"
        if not model_path.exists():
            model_path = model_dir / "final_model.zip"
        algo_cls = ALGOS[algo_name]
        model = algo_cls.load(model_path)

        eval_env = Monitor(gym.make(ENV_ID))
        set_global_seed(seed + 2000)
        eval_env.reset(seed=seed + 2000)
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True
        )
        rows.append(
            {
                "experiment": experiment,
                "variant": variant,
                "seed": seed,
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
                "n_eval_episodes": n_eval_episodes,
            }
        )
        eval_env.close()
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["exp1", "exp2"], default="exp1")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    parser.add_argument("--model_type", choices=["best", "final"], default="best")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_variant", default=None)
    parser.add_argument("--video_seed", type=int, default=None)
    args = parser.parse_args()

    experiments = get_experiments()
    configs = experiments[args.experiment]
    seeds = parse_seeds(args.seeds)

    root = Path(__file__).resolve().parents[1]
    model_root = root / "track1" / "artifacts" / "models"
    plot_root = root / "track1" / "artifacts" / "plots"
    video_root = root / "track1" / "artifacts" / "videos"

    all_rows = []
    for variant, config in configs.items():
        df = evaluate_variant(
            args.experiment,
            variant,
            config.algo,
            seeds,
            model_root,
            args.n_eval_episodes,
            args.model_type,
        )
        all_rows.append(df)

    results = pd.concat(all_rows, ignore_index=True)
    plot_root.mkdir(parents=True, exist_ok=True)
    results_path = plot_root / f"{args.experiment}_eval.csv"
    results.to_csv(results_path, index=False)

    summary = (
        results.groupby(["experiment", "variant"])
        .agg(mean_reward=("mean_reward", "mean"), std_reward=("mean_reward", "std"))
        .reset_index()
    )
    summary_path = plot_root / f"{args.experiment}_eval_summary.csv"
    summary.to_csv(summary_path, index=False)

    if args.record_video:
        if args.video_variant is None:
            best_variant = summary.sort_values("mean_reward", ascending=False).iloc[0][
                "variant"
            ]
        else:
            best_variant = args.video_variant

        variant_rows = results[results["variant"] == best_variant]
        if args.video_seed is None:
            best_seed = (
                variant_rows.sort_values("mean_reward", ascending=False)
                .iloc[0]["seed"]
            )
        else:
            best_seed = args.video_seed

        config = configs[best_variant]
        algo_cls = ALGOS[config.algo]
        model_path = (
            model_root / args.experiment / best_variant / f"seed_{int(best_seed)}"
        ) / f"{args.model_type}_model.zip"
        if not model_path.exists():
            model_path = model_path.with_name("final_model.zip")
        model = algo_cls.load(model_path)
        env = gym.make(ENV_ID, render_mode="rgb_array")
        video_root.mkdir(parents=True, exist_ok=True)
        out_path = video_root / f"{args.experiment}_{best_variant}_seed_{int(best_seed)}.mp4"
        record_episode(model, env, out_path, max_steps=200)
        env.close()


if __name__ == "__main__":
    main()
