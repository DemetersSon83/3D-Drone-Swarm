"""Importable, restart-safe execution of one drone-swarm scenario."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from drone_swarm.contracts import (
    AGENT_COLUMNS,
    AGENT_TEMPLATE,
    DATASET_CONTRACT_VERSION,
    EVENT_COLUMNS,
    EVENT_TEMPLATE,
    SWARM_TICK_COLUMNS,
    SWARM_TICK_TEMPLATE,
    build_run_artifact_manifest,
    preferred_resource_file,
)
from drone_swarm.io import (
    StreamingRecordWriter,
    StreamingTransitionWriter,
    atomic_write_json,
    write_records_csv,
    write_records_parquet,
)
from drone_swarm.projections import agent_signal_row, template_agent_signal_row
from drone_swarm.scenario import (
    build_model,
    load_scenario,
    make_run_id,
    resolve_run_seeds,
    scenario_config_hash,
    simulation_config_hash,
    validate_scenario,
)
from drone_swarm.validation import ValidationLevel, validate_run_directory


@dataclass(frozen=True, slots=True)
class RunOptions:
    config_path: Path
    output_root: Path
    base_seed: int
    initialization_seed: int | None = None
    policy_seed: int | None = None
    perturbation_seed: int | None = None
    run_id: str | None = None
    formats: tuple[str, ...] | None = None
    steps: int | None = None
    n_drones: int | None = None
    force: bool = False
    resume: bool = False
    write_agent_signals: bool | None = None
    validation_level: ValidationLevel = "standard"
    repository_root: Path | None = None
    command: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.force and self.resume:
            raise ValueError("force and resume are mutually exclusive")
        if self.base_seed < 0:
            raise ValueError("base_seed must be non-negative")
        if self.validation_level not in {"quick", "standard", "full"}:
            raise ValueError(f"unsupported validation level: {self.validation_level}")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _repository_root(start: Path | None = None) -> Path:
    candidate = (start or Path.cwd()).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file() and (directory / "src/drone_swarm").is_dir():
            return directory
    return candidate


def repository_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def repository_dirty(root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in (
        "drone-swarm-boids",
        "mesa",
        "numpy",
        "pandas",
        "pyarrow",
        "PyYAML",
        "duckdb",
        "polars",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def normalize_formats(config: dict[str, Any], override: tuple[str, ...] | None) -> tuple[str, ...]:
    if override:
        values = [str(item).strip().lower() for item in override if str(item).strip()]
    else:
        output = config.get("output", {})
        configured = output.get("formats", ["parquet"])
        values = [configured] if isinstance(configured, str) else list(configured)
        values = [str(item).lower() for item in values]
    normalized = tuple(dict.fromkeys(values))
    unsupported = set(normalized).difference({"parquet", "csv", "jsonl"})
    if unsupported:
        raise ValueError(f"unsupported output formats: {sorted(unsupported)}")
    if not normalized:
        raise ValueError("at least one output format is required")
    return normalized


def merge_agent_registry(
    initial_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    final_by_id = {row["agent_id"]: row for row in final_rows}
    merged: list[dict[str, Any]] = []
    for initial in initial_rows:
        final = final_by_id[initial["agent_id"]]
        row = {
            "agent_id": initial["agent_id"],
            "agent_index": initial["agent_index"],
        }
        for key, value in initial.items():
            if key not in {"agent_id", "agent_index"}:
                row[f"initial_{key}"] = value
        for key, value in final.items():
            if key not in {"agent_id", "agent_index"}:
                row[f"final_{key}"] = value
        merged.append(row)
    return merged


def write_table_variants(
    records: list[dict[str, Any]],
    directory: Path,
    stem: str,
    formats: tuple[str, ...],
    *,
    columns: tuple[str, ...] | None = None,
    template: dict[str, Any] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    if "parquet" in formats:
        path = directory / f"{stem}.parquet"
        write_records_parquet(records, path, columns=columns, template=template)
        paths.append(path)
    if "csv" in formats or "parquet" not in formats:
        path = directory / f"{stem}.csv"
        write_records_csv(records, path, columns=columns, template=template)
        paths.append(path)
    return paths


def _resource_path(
    artifacts: dict[str, Any],
    name: str,
    directory: Path,
) -> str | None:
    selected = preferred_resource_file(artifacts, name)
    if selected is None:
        return None
    return str((directory / str(selected["path"])).name)


def execute_run(options: RunOptions) -> dict[str, Any]:
    """Execute one scenario and return a machine-readable status mapping."""

    repository_root = _repository_root(options.repository_root or options.config_path)
    config = load_scenario(options.config_path)
    if options.steps is not None:
        config.setdefault("run", {})["steps"] = options.steps
    if options.n_drones is not None:
        config.setdefault("model", {})["n_drones"] = options.n_drones
    validate_scenario(config)

    seeds = resolve_run_seeds(
        options.base_seed,
        initialization_seed=options.initialization_seed,
        policy_seed=options.policy_seed,
        perturbation_seed=options.perturbation_seed,
    )
    run_id = options.run_id or make_run_id(config, seeds)
    formats = normalize_formats(config, options.formats)
    output_section = config.get("output", {})
    configured_projection = (
        bool(output_section.get("write_agent_signals", True))
        if isinstance(output_section, dict)
        else True
    )
    write_agent_signals = (
        configured_projection
        if options.write_agent_signals is None
        else bool(options.write_agent_signals)
    )

    output_root = options.output_root.resolve()
    raw_root = output_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    final_directory = raw_root / run_id
    success_marker = final_directory / "_SUCCESS"
    stale_partials = sorted(raw_root.glob(f".{run_id}.partial-*"))

    if stale_partials and not (options.force or options.resume):
        raise FileExistsError(
            f"incomplete partial run directories exist for {run_id}; use resume or force"
        )
    for stale_partial in stale_partials:
        shutil.rmtree(stale_partial)

    if success_marker.is_file() and not options.force:
        return {"status": "skipped", "run_id": run_id, "path": str(final_directory)}
    if final_directory.exists():
        if options.force or options.resume:
            shutil.rmtree(final_directory)
        else:
            raise FileExistsError(
                f"incomplete run directory exists: {final_directory}; use resume or force"
            )

    partial_directory = raw_root / f".{run_id}.partial-{os.getpid()}"
    if partial_directory.exists():
        shutil.rmtree(partial_directory)
    partial_directory.mkdir(parents=True)

    start_wall = time.perf_counter()
    started_at = utc_now()
    config_hash = scenario_config_hash(config)
    simulation_hash = simulation_config_hash(config)
    run_config = {
        "contract_version": DATASET_CONTRACT_VERSION,
        "run_id": run_id,
        "source_config": str(options.config_path.resolve()),
        "config_hash": config_hash,
        "simulation_hash": simulation_hash,
        "seeds": seeds.to_dict(),
        "scenario": config,
        "formats": formats,
        "write_agent_signals": write_agent_signals,
    }
    atomic_write_json(partial_directory / "config.json", run_config)

    steps = int(config["run"]["steps"])
    metrics_stride = int(config["run"].get("metrics_stride", 10))
    batch_size = int(config["run"].get("transition_batch_size", 20_000))
    progress_interval = int(config["run"].get("progress_interval", 0))
    metrics_rows: list[dict[str, Any]] = []
    command = list(options.command) if options.command else list(sys.argv)

    transition_writer: StreamingTransitionWriter | None = None
    signals_writer: StreamingRecordWriter | None = None
    try:
        transition_writer = StreamingTransitionWriter(
            partial_directory,
            formats=formats,
            batch_size=batch_size,
        )
        projection_formats = tuple(value for value in formats if value in {"parquet", "csv"})
        signal_context = {
            "scenario_id": config["scenario_id"],
            "analysis_split": config.get("analysis_split", "unspecified"),
            "base_seed": seeds.base_seed,
            "initialization_seed": seeds.initialization_seed,
            "policy_seed": seeds.policy_seed,
            "perturbation_seed": seeds.perturbation_seed,
            "dt": float(config["model"].get("dt", 0.25)),
        }
        if write_agent_signals and projection_formats:
            signals_writer = StreamingRecordWriter(
                partial_directory,
                stem="agent_signals",
                template=template_agent_signal_row(),
                formats=projection_formats,
                batch_size=batch_size,
            )

        def record_transition(transition: Any) -> None:
            assert transition_writer is not None
            transition_writer.write(transition)
            if signals_writer is not None:
                signals_writer.write(agent_signal_row(transition, context=signal_context))

        model = build_model(
            config,
            seeds=seeds,
            run_id=run_id,
            transition_callback=record_transition,
            retain_transitions=False,
        )
        initial_registry = model.agent_registry()

        for completed in range(1, steps + 1):
            model.step()
            if completed % metrics_stride == 0 or completed == steps:
                row = model.metrics_snapshot()
                row["run_id"] = run_id
                row["scenario_id"] = config["scenario_id"]
                row["base_seed"] = seeds.base_seed
                row["phase"] = model.event_manager.phase_for_step(max(model.tick - 1, 0))
                metrics_rows.append(row)
            if progress_interval and completed % progress_interval == 0:
                print(
                    json.dumps(
                        {
                            "status": "running",
                            "run_id": run_id,
                            "completed_steps": completed,
                            "total_steps": steps,
                        }
                    ),
                    flush=True,
                )

        transition_writer.close()
        if signals_writer is not None:
            signals_writer.close()
        expected_rows = int(config["model"].get("n_drones", 48)) * steps
        observed_signal_rows = signals_writer.row_count if signals_writer is not None else None
        if transition_writer.row_count != expected_rows or model.transition_count != expected_rows:
            raise RuntimeError(
                "transition row-count validation failed: "
                f"writer={transition_writer.row_count}, model={model.transition_count}, "
                f"expected={expected_rows}"
            )
        if observed_signal_rows is not None and observed_signal_rows != expected_rows:
            raise RuntimeError(
                f"agent_signals row-count validation failed: {observed_signal_rows} != {expected_rows}"
            )

        event_records = model.event_manager.event_records()
        for record in event_records:
            record["run_id"] = run_id
            record["scenario_id"] = config["scenario_id"]
            record["base_seed"] = seeds.base_seed
        final_registry = model.agent_registry()
        agent_records = merge_agent_registry(initial_registry, final_registry)
        for record in agent_records:
            record["run_id"] = run_id
            record["scenario_id"] = config["scenario_id"]
            record["base_seed"] = seeds.base_seed

        write_table_variants(
            event_records,
            partial_directory,
            "events",
            formats,
            columns=EVENT_COLUMNS,
            template=EVENT_TEMPLATE,
        )
        write_table_variants(
            metrics_rows,
            partial_directory,
            "swarm_ticks",
            formats,
            columns=SWARM_TICK_COLUMNS,
            template=SWARM_TICK_TEMPLATE,
        )
        write_table_variants(
            agent_records,
            partial_directory,
            "agents",
            formats,
            columns=AGENT_COLUMNS,
            template=AGENT_TEMPLATE,
        )
        atomic_write_json(partial_directory / "events.json", event_records)

        artifacts = build_run_artifact_manifest(
            partial_directory,
            run_id=run_id,
            scenario_id=str(config["scenario_id"]),
        )
        atomic_write_json(partial_directory / "artifact_manifest.json", artifacts)

        ended_at = utc_now()
        runtime_seconds = time.perf_counter() - start_wall
        provenance = {
            "contract_version": DATASET_CONTRACT_VERSION,
            "run_id": run_id,
            "scenario_id": config["scenario_id"],
            "config_hash": config_hash,
            "simulation_hash": simulation_hash,
            "repository_commit": repository_commit(repository_root),
            "repository_dirty": repository_dirty(repository_root),
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "runtime_seconds": runtime_seconds,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "packages": package_versions(),
            "command": command,
            "seeds": seeds.to_dict(),
            "resources": artifacts["resources"],
        }
        atomic_write_json(partial_directory / "provenance.json", provenance)

        event_types = [record["event_type"] for record in event_records]
        manifest_row = {
            "contract_version": DATASET_CONTRACT_VERSION,
            "run_id": run_id,
            "scenario_id": config["scenario_id"],
            "analysis_split": config.get("analysis_split", "unspecified"),
            "paired_control_scenario": config.get("paired_control_scenario"),
            "paired_treatment_scenario": config.get("paired_treatment_scenario"),
            "config_hash": config_hash,
            "simulation_hash": simulation_hash,
            "repository_commit": provenance["repository_commit"],
            "base_seed": seeds.base_seed,
            "initialization_seed": seeds.initialization_seed,
            "policy_seed": seeds.policy_seed,
            "perturbation_seed": seeds.perturbation_seed,
            "n_drones": int(config["model"].get("n_drones", 48)),
            "steps": steps,
            "dt": float(config["model"].get("dt", 0.25)),
            "activation": str(config["model"].get("activation", "staged")),
            "event_types": json.dumps(event_types, separators=(",", ":")),
            "event_count": len(event_records),
            "transition_rows": transition_writer.row_count,
            "agent_signal_rows": observed_signal_rows,
            "metrics_rows": len(metrics_rows),
            "formats": json.dumps(formats, separators=(",", ":")),
            "transitions_file": _resource_path(artifacts, "transitions", partial_directory),
            "agent_signals_file": _resource_path(artifacts, "agent_signals", partial_directory),
            "swarm_ticks_file": _resource_path(artifacts, "swarm_ticks", partial_directory),
            "agents_file": _resource_path(artifacts, "agents", partial_directory),
            "events_file": _resource_path(artifacts, "events", partial_directory),
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "runtime_seconds": runtime_seconds,
            "status": "success",
        }
        atomic_write_json(partial_directory / "manifest_row.json", manifest_row)

        quality = validate_run_directory(
            partial_directory,
            level=options.validation_level,
            require_success=False,
            verify_checksums=True,
        )
        # The checks run against the hidden staging directory, but the report is
        # part of the finalized public run. Record its durable location rather
        # than an ephemeral ``.partial-*`` path.
        quality["path"] = str(final_directory)
        quality["validation_stage"] = "pre_atomic_finalize"
        atomic_write_json(partial_directory / "quality_report.json", quality)
        if quality["status"] == "fail":
            raise RuntimeError(
                f"run artifact validation failed with {quality['error_count']} error(s)"
            )

        atomic_write_json(
            partial_directory / "_SUCCESS",
            {
                "contract_version": DATASET_CONTRACT_VERSION,
                "run_id": run_id,
                "transition_rows": transition_writer.row_count,
                "agent_signal_rows": observed_signal_rows,
                "quality_status": quality["status"],
                "completed_at_utc": ended_at,
            },
        )
        os.replace(partial_directory, final_directory)
        return {
            "status": "success",
            "run_id": run_id,
            "path": str(final_directory),
            "transition_rows": transition_writer.row_count,
            "agent_signal_rows": observed_signal_rows,
            "quality_status": quality["status"],
            "runtime_seconds": runtime_seconds,
        }
    except Exception as exc:
        if transition_writer is not None:
            with contextlib.suppress(Exception):
                transition_writer.close()
        if signals_writer is not None:
            with contextlib.suppress(Exception):
                signals_writer.close()
        atomic_write_json(
            partial_directory / "failure.json",
            {
                "run_id": run_id,
                "failed_at_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "command": command,
            },
        )
        raise
