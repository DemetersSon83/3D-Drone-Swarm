"""Run a basic 3D drone swarm simulation and export transition data."""

from __future__ import annotations

import argparse
from pathlib import Path

from drone_swarm.model import DroneSwarmModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation ticks to run.")
    parser.add_argument("--n-drones", type=int, default=50, help="Number of drones to simulate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/transitions.csv"),
        help="CSV file for flattened transition records.",
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=None,
        help="Optional JSONL file for nested transition records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = DroneSwarmModel(
        n_drones=args.n_drones,
        bounds=((0.0, 100.0), (0.0, 100.0), (0.0, 50.0)),
        seed=args.seed,
        dt=1.0,
        perception_radius=12.0,
        separation_distance=3.0,
        collision_radius=1.0,
        max_speed=3.0,
        max_acceleration=0.5,
        boundary_mode="bounce",
        boundary_margin=7.5,
        activation="staged",
    )
    model.run_steps(args.steps)

    model.export_transitions_csv(str(args.output))
    if args.jsonl_output is not None:
        model.export_transitions_jsonl(str(args.jsonl_output))

    df = model.transitions_dataframe()
    print(f"Ran {args.steps} steps with {args.n_drones} drones.")
    print(f"Wrote {len(df)} transition rows to {args.output}.")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
