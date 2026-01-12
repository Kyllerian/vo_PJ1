from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from track1.configs import SEEDS, get_experiments
from utils.plotting import plot_groups


def parse_seeds(seeds_str: str) -> List[int]:
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["exp1", "exp2"], default="exp1")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--smoothing_window", type=int, default=50)
    parser.add_argument("--n_points", type=int, default=200)
    args = parser.parse_args()

    experiments = get_experiments()
    configs = experiments[args.experiment]
    seeds = parse_seeds(args.seeds)

    root = Path(__file__).resolve().parents[1]
    log_root = root / "track1" / "artifacts" / "logs"
    plot_root = root / "track1" / "artifacts" / "plots"

    groups: Dict[str, List[Path]] = {}
    for variant in configs.keys():
        dirs = []
        for seed in seeds:
            run_dir = log_root / args.experiment / variant / f"seed_{seed}"
            dirs.append(run_dir)
        groups[variant] = dirs

    out_path = plot_root / f"{args.experiment}_learning_curve.png"
    plot_groups(
        groups,
        out_path,
        title=f"Track1 {args.experiment} learning curves",
        smoothing_window=args.smoothing_window,
        n_points=args.n_points,
    )


if __name__ == "__main__":
    main()
