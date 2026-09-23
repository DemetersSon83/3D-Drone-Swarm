"""Command-line interface for simulation, cataloging, and artifact validation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from drone_swarm import __version__
from drone_swarm.dataset import build_dataset_catalog
from drone_swarm.runner import RunOptions, execute_run
from drone_swarm.scenario import load_scenario, make_run_id, resolve_run_seeds, validate_scenario
from drone_swarm.schema_resources import load_schema, write_schema
from drone_swarm.validation import validate_run_directory


def _formats(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    result = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("formats must contain at least one value")
    return result


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="YAML or JSON scenario file.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/swarm_dataset_v1"),
        help="Dataset root; runs are written under raw/<run_id>/.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Paired-run base seed.")
    parser.add_argument("--initialization-seed", type=int, default=None)
    parser.add_argument("--policy-seed", type=int, default=None)
    parser.add_argument("--perturbation-seed", type=int, default=None)
    parser.add_argument("--run-id", default=None, help="Override the deterministic run ID.")
    parser.add_argument(
        "--formats",
        default=None,
        help="Comma-separated transition formats overriding output.formats.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override run.steps.")
    parser.add_argument("--n-drones", type=int, default=None, help="Override model.n_drones.")
    parser.add_argument("--force", action="store_true", help="Replace an existing completed run.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed runs and restart incomplete runs from scratch.",
    )
    signal_group = parser.add_mutually_exclusive_group()
    signal_group.add_argument(
        "--agent-signals",
        dest="agent_signals",
        action="store_true",
        help="Force generation of the compact analysis projection.",
    )
    signal_group.add_argument(
        "--no-agent-signals",
        dest="agent_signals",
        action="store_false",
        help="Disable the compact analysis projection.",
    )
    parser.set_defaults(agent_signals=None)
    parser.add_argument(
        "--validation-level",
        choices=("quick", "standard", "full"),
        default="standard",
        help="Run-level artifact validation depth.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drone-swarm")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one reproducible scenario.")
    _add_run_arguments(run_parser)

    id_parser = subparsers.add_parser("run-id", help="Resolve and print a deterministic run ID.")
    id_parser.add_argument("--config", type=Path, required=True)
    id_parser.add_argument("--seed", type=int, required=True)
    id_parser.add_argument("--initialization-seed", type=int, default=None)
    id_parser.add_argument("--policy-seed", type=int, default=None)
    id_parser.add_argument("--perturbation-seed", type=int, default=None)

    validate_scenario_parser = subparsers.add_parser(
        "validate-scenario", help="Validate a scenario configuration without running it."
    )
    validate_scenario_parser.add_argument("--config", type=Path, required=True)

    catalog_parser = subparsers.add_parser(
        "catalog", help="Build dataset-level manifests, schemas, SQL views, and checksums."
    )
    catalog_parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/swarm_dataset_v1")
    )
    catalog_parser.add_argument("--strict", action="store_true")
    catalog_parser.add_argument("--no-parquet", action="store_true")
    catalog_parser.add_argument("--skip-run-validation", action="store_true")
    catalog_parser.add_argument(
        "--validation-level", choices=("quick", "standard", "full"), default="quick"
    )

    validate_run_parser = subparsers.add_parser(
        "validate-run", help="Validate an existing run directory."
    )
    validate_run_parser.add_argument("run_directory", type=Path)
    validate_run_parser.add_argument(
        "--level", choices=("quick", "standard", "full"), default="standard"
    )
    validate_run_parser.add_argument("--no-checksums", action="store_true")

    schema_parser = subparsers.add_parser(
        "show-schema", help="Print or write one bundled machine-readable JSON schema."
    )
    schema_parser.add_argument("name", choices=("scenario", "artifact-manifest", "dataset-catalog"))
    schema_parser.add_argument("--output", type=Path, default=None)

    replay_parser = subparsers.add_parser(
        "replay", help="Replay observed pre-state kinematics on an explicit causal research clock."
    )
    replay_parser.add_argument("run_directory", type=Path)
    replay_parser.add_argument("--output-directory", type=Path, required=True)
    replay_parser.add_argument(
        "--availability-basis",
        choices=("completed_transition_lower_bound", "idealized_pre_state_replay"),
        default="completed_transition_lower_bound",
    )
    replay_parser.add_argument("--frame-id", default="simulation_cartesian")
    replay_parser.add_argument("--frame-origin", default="unspecified_legacy_simulation")
    replay_parser.add_argument("--frame-handedness", default="unspecified_legacy_simulation")
    replay_parser.add_argument("--max-tracks", type=int, default=128)
    replay_parser.add_argument("--max-queued-frames", type=int, default=4)
    replay_parser.add_argument("--allowed-lateness", type=float, default=None)
    replay_parser.add_argument("--batch-size", type=int, default=1024)

    observations_parser = subparsers.add_parser(
        "validate-observations", help="Validate replay JSONL records and causal frame admission."
    )
    observations_parser.add_argument("input_file", type=Path)
    observations_parser.add_argument("--metadata", type=Path, required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "replay":
        from drone_swarm.replay import ReplayOptions, replay_run

        try:
            result = replay_run(
                ReplayOptions(
                    run_directory=args.run_directory,
                    output_directory=args.output_directory,
                    availability_basis=args.availability_basis,
                    frame_id=args.frame_id,
                    frame_origin=args.frame_origin,
                    frame_handedness=args.frame_handedness,
                    max_tracks=args.max_tracks,
                    max_queued_frames=args.max_queued_frames,
                    allowed_lateness=args.allowed_lateness,
                    batch_size=args.batch_size,
                )
            )
        except (OSError, ValueError, RuntimeError, csv.Error) as exc:
            print(json.dumps({"status": "fail", "error": str(exc)}), file=sys.stderr)
            return 1
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "pass" else 1

    if args.command == "validate-observations":
        from drone_swarm.replay import validate_observation_file

        report = validate_observation_file(args.input_file, metadata_path=args.metadata)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "pass" else 1

    if args.command == "run":
        if args.force and args.resume:
            parser.error("--force and --resume are mutually exclusive")
        formats = _formats(args.formats)
        result = execute_run(
            RunOptions(
                config_path=args.config,
                output_root=args.output_root,
                base_seed=args.seed,
                initialization_seed=args.initialization_seed,
                policy_seed=args.policy_seed,
                perturbation_seed=args.perturbation_seed,
                run_id=args.run_id,
                formats=formats,
                steps=args.steps,
                n_drones=args.n_drones,
                force=args.force,
                resume=args.resume,
                write_agent_signals=args.agent_signals,
                validation_level=args.validation_level,
                command=tuple(sys.argv if argv is None else ("drone-swarm", *argv)),
            )
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "run-id":
        config = load_scenario(args.config)
        seeds = resolve_run_seeds(
            args.seed,
            initialization_seed=args.initialization_seed,
            policy_seed=args.policy_seed,
            perturbation_seed=args.perturbation_seed,
        )
        print(make_run_id(config, seeds))
        return 0

    if args.command == "validate-scenario":
        config = load_scenario(args.config)
        validate_scenario(config)
        print(json.dumps({"status": "valid", "scenario_id": config["scenario_id"]}))
        return 0

    if args.command == "catalog":
        summary = build_dataset_catalog(
            args.output_root,
            strict=args.strict,
            no_parquet=args.no_parquet,
            validate_runs=not args.skip_run_validation,
            validation_level=args.validation_level,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0 if summary.get("quality_status") != "fail" else 1

    if args.command == "validate-run":
        report = validate_run_directory(
            args.run_directory,
            level=args.level,
            verify_checksums=not args.no_checksums,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] != "fail" else 1

    if args.command == "show-schema":
        if args.output is not None:
            output = write_schema(args.name, args.output)
            print(json.dumps({"status": "written", "path": str(output)}))
        else:
            print(json.dumps(load_schema(args.name), indent=2, sort_keys=True))
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
