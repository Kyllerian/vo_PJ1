from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from track2.configs import SEEDS, get_configs
from track2.envs.gridworld import GridWorldEnv
from utils.seed import set_global_seed
from utils.video import record_episode


def parse_seeds(seeds_str: str) -> List[int]:
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]


def evaluate_model(
    model: DQN,
    reward_mode: str,
    n_eval_episodes: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    env = GridWorldEnv(reward_mode=reward_mode)
    set_global_seed(seed)
    rewards = []
    lengths = []
    successes = []
    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        ep_len = 0
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            ep_len += 1
        rewards.append(total_reward)
        lengths.append(ep_len)
        successes.append(bool(info.get("is_success", False)))
    env.close()
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    success_rate = float(np.mean(successes))
    mean_length = float(np.mean(lengths))
    return mean_reward, std_reward, success_rate, mean_length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="sparse,dense")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_variant", default=None)
    parser.add_argument("--video_seed", type=int, default=None)
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = parse_seeds(args.seeds)

    root = Path(__file__).resolve().parents[1]
    model_root = root / "track2" / "artifacts" / "models"
    plot_root = root / "track2" / "artifacts" / "plots"
    video_root = root / "track2" / "artifacts" / "videos"

    rows = []
    for variant in variants:
        for seed in seeds:
            model_path = model_root / variant / f"seed_{seed}" / "final_model.zip"
            model = DQN.load(model_path)
            mean_reward, std_reward, success_rate, mean_length = evaluate_model(
                model, variant, args.n_eval_episodes, seed + 2000
            )
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                    "success_rate": success_rate,
                    "mean_length": mean_length,
                    "n_eval_episodes": args.n_eval_episodes,
                }
            )

    results = pd.DataFrame(rows)
    plot_root.mkdir(parents=True, exist_ok=True)
    results_path = plot_root / "track2_eval.csv"
    results.to_csv(results_path, index=False)

    summary = (
        results.groupby("variant")
        .agg(
            mean_reward=("mean_reward", "mean"),
            std_reward=("mean_reward", "std"),
            success_rate=("success_rate", "mean"),
            mean_length=("mean_length", "mean"),
        )
        .reset_index()
    )
    summary_path = plot_root / "track2_eval_summary.csv"
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

        model_path = (
            model_root / best_variant / f"seed_{int(best_seed)}" / "final_model.zip"
        )
        model = DQN.load(model_path)
        env = GridWorldEnv(reward_mode=best_variant, render_mode="rgb_array")
        video_root.mkdir(parents=True, exist_ok=True)
        out_path = video_root / f"track2_{best_variant}_seed_{int(best_seed)}.mp4"
        record_episode(model, env, out_path, max_steps=100)
        env.close()


if __name__ == "__main__":
    main()
