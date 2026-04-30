"""Plot trajectories from an exported transition CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from drone_swarm.viz3d import plot_transition_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV file produced by run_basic_swarm.py.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/trajectories.png"),
        help="Output image path.",
    )
    parser.add_argument("--max-agents", type=int, default=12, help="Maximum agents to plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    plot_transition_trajectories(df, output_path=args.output, max_agents=args.max_agents)
    print(f"Wrote trajectory plot to {args.output}.")


if __name__ == "__main__":
    main()
