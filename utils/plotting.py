from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class RunSeries:
    steps: np.ndarray
    rewards: np.ndarray


def _find_monitor_file(log_dir: Path) -> Path:
    candidates = list(log_dir.glob("*.monitor.csv"))
    if not candidates:
        candidates = list(log_dir.glob("monitor.csv"))
    if not candidates:
        raise FileNotFoundError(f"No monitor file in {log_dir}")
    return candidates[0]


def load_monitor(log_dir: Path) -> RunSeries:
    monitor_path = _find_monitor_file(log_dir)
    df = pd.read_csv(monitor_path, comment="#")
    if "l" in df.columns:
        steps = df["l"].cumsum().to_numpy()
    else:
        steps = np.arange(len(df), dtype=np.int64)
    rewards = df["r"].to_numpy()
    return RunSeries(steps=steps, rewards=rewards)


def smooth_rewards(rewards: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return rewards
    series = pd.Series(rewards)
    return series.rolling(window=window, min_periods=1).mean().to_numpy()


def aggregate_runs(
    log_dirs: Iterable[Path],
    smoothing_window: int = 50,
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = []
    max_steps = None
    for log_dir in log_dirs:
        run = load_monitor(Path(log_dir))
        run_rewards = smooth_rewards(run.rewards, smoothing_window)
        series.append(RunSeries(steps=run.steps, rewards=run_rewards))
        run_max = run.steps.max() if len(run.steps) else 0
        max_steps = run_max if max_steps is None else min(max_steps, run_max)
    if max_steps is None:
        raise ValueError("No runs provided.")
    x = np.linspace(0, max_steps, n_points)
    ys = []
    for run in series:
        ys.append(np.interp(x, run.steps, run.rewards))
    y = np.vstack(ys)
    return x, y.mean(axis=0), y.std(axis=0)


def plot_groups(
    groups: Dict[str, List[Path]],
    out_path: Path,
    title: str,
    smoothing_window: int = 50,
    n_points: int = 200,
    ylabel: str = "Episode reward (smoothed)",
) -> None:
    plt.figure(figsize=(8, 5))
    for label, dirs in groups.items():
        x, mean, std = aggregate_runs(
            dirs, smoothing_window=smoothing_window, n_points=n_points
        )
        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
