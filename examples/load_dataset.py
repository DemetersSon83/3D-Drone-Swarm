#!/usr/bin/env python3
"""Load a generated swarm dataset and print a compact analytical summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from drone_swarm.loaders import DroneSwarmDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--limit-runs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = DroneSwarmDataset(args.dataset_root)
    manifest = dataset.manifest_pandas()
    if args.scenario:
        manifest = manifest[manifest["scenario_id"].isin(args.scenario)]
    selected = manifest.head(max(args.limit_runs, 0))
    print(
        selected[["run_id", "scenario_id", "base_seed", "transition_rows"]].to_string(index=False)
    )
    if selected.empty:
        return
    signals = dataset.read_pandas(
        "agent_signals",
        scenario_ids=selected["scenario_id"].unique(),
        seeds=selected["base_seed"].unique(),
        columns=[
            "run_id",
            "scenario_id",
            "base_seed",
            "step",
            "agent_index",
            "phase",
            "coalition_truth",
            "observed_state_speed",
            "observed_state_neighbor_count",
        ],
    )
    selected_ids = set(selected["run_id"])
    signals = signals[signals["run_id"].isin(selected_ids)]
    print()
    print(
        signals.groupby(["scenario_id", "base_seed", "phase"], dropna=False)
        .agg(
            rows=("step", "size"),
            mean_speed=("observed_state_speed", "mean"),
            mean_neighbors=("observed_state_neighbor_count", "mean"),
        )
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
