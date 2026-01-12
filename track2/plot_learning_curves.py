from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from track2.configs import SEEDS
from utils.plotting import plot_groups


def parse_seeds(seeds_str: str) -> List[int]:
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="sparse,dense")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--smoothing_window", type=int, default=20)
    parser.add_argument("--n_points", type=int, default=200)
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = parse_seeds(args.seeds)

    root = Path(__file__).resolve().parents[1]
    log_root = root / "track2" / "artifacts" / "logs"
    plot_root = root / "track2" / "artifacts" / "plots"

    groups: Dict[str, List[Path]] = {}
    for variant in variants:
        dirs = []
        for seed in seeds:
            run_dir = log_root / variant / f"seed_{seed}"
            dirs.append(run_dir)
        groups[variant] = dirs

    out_path = plot_root / "track2_learning_curve.png"
    plot_groups(
        groups,
        out_path,
        title="Track2 GridWorld learning curves",
        smoothing_window=args.smoothing_window,
        n_points=args.n_points,
    )


if __name__ == "__main__":
    main()
