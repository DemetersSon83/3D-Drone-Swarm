"""Plot trajectories from an exported transition CSV or Parquet table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from drone_swarm.viz3d import plot_transition_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path, help="Transition or agent-signals CSV/Parquet file.")
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
    if args.table.suffix.lower() == ".parquet":
        frame = pd.read_parquet(args.table)
    else:
        frame = pd.read_csv(args.table)
    plot_transition_trajectories(frame, output_path=args.output, max_agents=args.max_agents)
    print(f"Wrote trajectory plot to {args.output}.")


if __name__ == "__main__":
    main()
