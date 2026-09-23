"""Causal, bounded replay of the simulator's pre-state observation projection.

This is an offline research clock, not a paced sensor driver. No future state,
intervention label, control, or planted metadata enters observation payloads.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from drone_swarm import __version__
from drone_swarm.contracts import AGENT_SIGNALS_REQUIRED_COLUMNS, preferred_resource_file
from drone_swarm.io import atomic_write_json, sha256_file
from drone_swarm.observations import Observation, ObservationError, observation_from_signal
from drone_swarm.replay_frames import FrameAssembler
from drone_swarm.validation import validate_run_directory

REPLAY_VERSION = "0.1.0"
AVAILABILITY_BASES = ("completed_transition_lower_bound", "idealized_pre_state_replay")
SIGNAL_COLUMNS = (
    "contract_version",
    "run_id",
    "step",
    "time",
    "next_time",
    "agent_id",
    "observed_state_position_x",
    "observed_state_position_y",
    "observed_state_position_z",
    "observed_state_velocity_x",
    "observed_state_velocity_y",
    "observed_state_velocity_z",
)
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_RECORD_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ReplayOptions:
    run_directory: Path
    output_directory: Path
    availability_basis: str = "completed_transition_lower_bound"
    frame_id: str = "simulation_cartesian"
    frame_origin: str = "unspecified_legacy_simulation"
    frame_handedness: str = "unspecified_legacy_simulation"
    max_tracks: int = 128
    max_queued_frames: int = 4
    allowed_lateness: float | None = None
    batch_size: int = 1024


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if str(parsed) != value.strip():
            raise ValueError(f"{name} must be a canonical integer")
        value = parsed
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return result


def _json_object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"metadata exceeds {MAX_JSON_BYTES} bytes: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_keys)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _unique_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _within(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("artifact path must be relative to its run directory")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"artifact path is outside the run or missing: {relative}")
    return path


def _declared_columns(file_info: Mapping[str, Any]) -> list[str]:
    columns = file_info.get("columns")
    if not isinstance(columns, list):
        raise ValueError("artifact is missing its declared column schema")
    names = [c.get("name") if isinstance(c, dict) else None for c in columns]
    if not all(isinstance(n, str) and n for n in names) or len(set(names)) != len(names):
        raise ValueError("artifact column names must be unique nonempty strings")
    return [str(n) for n in names]


def _rows(
    path: Path,
    file_info: Mapping[str, Any],
    selected: Sequence[str],
    *,
    batch_size: int,
) -> Iterator[dict[str, Any]]:
    """Project bounded batches; never materialize the full transition table."""
    declared = _declared_columns(file_info)
    if not set(selected).issubset(declared):
        raise ValueError(f"required observation columns missing: {path.name}")
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != declared:
                raise ValueError(f"CSV header disagrees with artifact schema: {path.name}")
            for row in reader:
                if None in row or any(row[name] is None for name in selected):
                    raise ValueError(f"malformed CSV row: {path.name}")
                yield {name: row[name] for name in selected}
    elif path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ValueError("Parquet replay requires the package's parquet extra") from exc
        parquet = pq.ParquetFile(path)
        try:
            if parquet.schema_arrow.names != declared:
                raise ValueError(f"Parquet header disagrees with artifact schema: {path.name}")
            for batch in parquet.iter_batches(batch_size=batch_size, columns=list(selected)):
                yield from batch.to_pylist()
        finally:
            parquet.close()
    else:
        raise ValueError("replay accepts CSV or Parquet source tables")


def _metadata(options: ReplayOptions) -> tuple[dict[str, Any], Path, Mapping[str, Any]]:
    root = Path(options.run_directory).resolve()
    if options.availability_basis not in AVAILABILITY_BASES:
        raise ValueError("unsupported availability basis")
    for name in ("frame_id", "frame_origin", "frame_handedness"):
        if not isinstance(getattr(options, name), str) or not getattr(options, name).strip():
            raise ValueError(f"{name} must be explicitly declared")
    _integer(options.max_tracks, "max_tracks", minimum=1)
    _integer(options.max_queued_frames, "max_queued_frames", minimum=1)
    _integer(options.batch_size, "batch_size", minimum=1)
    if options.batch_size > 65536:
        raise ValueError("batch_size exceeds 65536")
    artifacts = _json_object(root / "artifact_manifest.json")
    config = _json_object(root / "config.json")
    manifest = _json_object(root / "manifest_row.json")
    _json_object(root / "provenance.json")
    resources = artifacts.get("resources")
    if not isinstance(resources, list) or not resources:
        raise ValueError("artifact manifest must contain resources")
    for resource in resources:
        if not isinstance(resource, dict) or not isinstance(resource.get("files"), list):
            raise ValueError("malformed resource entry")
        for file_info in resource["files"]:
            if not isinstance(file_info, dict):
                raise ValueError("malformed file entry")
            _within(root, file_info.get("path"))
    try:
        report = validate_run_directory(root, level="standard", verify_checksums=True)
    except (TypeError, KeyError, ArithmeticError) as exc:
        raise ValueError(f"malformed source metadata: {exc}") from exc
    if report["status"] == "fail":
        failures = [c["check"] for c in report["checks"] if not c["passed"]]
        raise ValueError("source validation failed: " + ", ".join(failures))
    success = _json_object(root / "_SUCCESS")
    if success.get("quality_status") not in ("pass", "warning"):
        raise ValueError("source completion marker lacks acceptable quality status")
    quality_path = root / "quality_report.json"
    if quality_path.exists() and _json_object(quality_path).get("status") not in (
        "pass",
        "warning",
    ):
        raise ValueError("source quality report failed")
    scenario = config.get("scenario")
    if not isinstance(scenario, dict) or not isinstance(scenario.get("model"), dict):
        raise ValueError("source config lacks scenario.model")
    model = scenario["model"]
    activation = model.get("activation", manifest.get("activation"))
    if activation != "staged" or manifest.get("activation") != "staged":
        raise ValueError("replay requires explicitly staged simulator activation")
    dt = _positive(model.get("dt"), "dt")
    if not math.isclose(dt, _positive(manifest.get("dt"), "manifest dt"), rel_tol=1e-12):
        raise ValueError("source dt declarations disagree")
    n = _integer(model.get("n_drones"), "n_drones", minimum=1)
    if n > options.max_tracks or n != _integer(manifest.get("n_drones"), "manifest n_drones"):
        raise ValueError("source population exceeds capacity or declarations disagree")
    steps = _integer(manifest.get("steps"), "steps", minimum=1)
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("source run_id missing")
    signals = preferred_resource_file(artifacts, "agent_signals", preference=("parquet", "csv"))
    registry = preferred_resource_file(artifacts, "agents", preference=("parquet", "csv"))
    if signals is None or registry is None:
        raise ValueError("replay requires agent_signals and agents tables")
    if not set(AGENT_SIGNALS_REQUIRED_COLUMNS).issubset(_declared_columns(signals)):
        raise ValueError("agent_signals schema is missing required contract columns")
    ids: list[str] = []
    for row in _rows(
        _within(root, registry["path"]),
        registry,
        ("run_id", "agent_id"),
        batch_size=options.batch_size,
    ):
        if row["run_id"] != run_id or len(ids) >= options.max_tracks:
            raise ValueError("invalid or over-capacity agent registry")
        ids.append(str(_integer(row["agent_id"], "agent_id")))
    if len(ids) != n or len(set(ids)) != n:
        raise ValueError("agent registry must contain exactly the declared unique cohort")
    signal_path = _within(root, signals["path"])
    lateness = (
        dt
        if options.allowed_lateness is None
        else _positive(options.allowed_lateness, "allowed_lateness", allow_zero=True)
    )
    metadata: dict[str, Any] = {
        "replay_version": REPLAY_VERSION,
        "observation_contract_version": REPLAY_VERSION,
        "producer_version": __version__,
        "stream_id": run_id,
        "source_id": "drone_swarm.agent_signals",
        "run_id": run_id,
        "source_sha256": signals["sha256"],
        "source_schema_sha256": hashlib.sha256(
            json.dumps(signals["columns"], sort_keys=True).encode()
        ).hexdigest(),
        "source_repository_commit": manifest.get("repository_commit"),
        "source_table": signal_path.name,
        "sample_id_policy": "json_tuple_stream_step_track_pre_state",
        "source_time_roundoff_policy": "validate_then_normalize_to_declared_step_grid",
        "source_validation_status": report["status"],
        "track_ids": sorted(ids, key=int),
        "expected_steps": steps,
        "dt": dt,
        "availability_basis": options.availability_basis,
        "clock_id": "simulation",
        "time_unit": "simulation_time",
        "length_unit": "simulation_length",
        "frame_id": options.frame_id,
        "frame_origin": options.frame_origin,
        "frame_handedness": options.frame_handedness,
        "axis_order": ["x", "y", "z"],
        "boundary_convention": model.get("boundary_mode", model.get("boundary", "unspecified")),
        "activation": activation,
        "measured_delivery_latency": False,
        "physical_unit_calibration": None,
        "covariance_calibration": None,
        "clock_uncertainty": None,
        "max_tracks": options.max_tracks,
        "max_queued_frames": options.max_queued_frames,
        "allowed_lateness": lateness,
        "lateness_reference": "nominal_available_time",
        "replay_pacing": "unpaced_logical_clock",
        "input_samples_per_simulation_time_unit": 1.0 / dt,
        "population_semantics": "fixed_simulated_cohort_not_external_population_truth",
    }
    return metadata, signal_path, signals


def _assembler(metadata: Mapping[str, Any]) -> FrameAssembler:
    if not isinstance(metadata.get("track_ids"), list):
        raise ValueError("metadata track_ids must be an array")
    if metadata.get("lateness_reference") != "nominal_available_time":
        raise ValueError("metadata must reference nominal availability for lateness")
    if metadata.get("sample_id_policy") != "json_tuple_stream_step_track_pre_state":
        raise ValueError("metadata sample ID policy is unsupported")
    return FrameAssembler(
        expected_track_ids=tuple(metadata["track_ids"]),
        dt=metadata["dt"],
        stream_id=metadata["stream_id"],
        availability_basis=metadata["availability_basis"],
        allowed_lateness=metadata["allowed_lateness"],
        max_queued_frames=metadata["max_queued_frames"],
        max_tracks=metadata["max_tracks"],
        expected_steps=metadata["expected_steps"],
    )


def _write_line(file: TextIO, value: Mapping[str, Any]) -> None:
    file.write(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")


def _status(summary: Mapping[str, Any], extra_errors: int = 0) -> str:
    failures = ("frames_skipped", "observations_rejected", "late_records", "history_resets")
    return "fail" if extra_errors or any(summary.get(key, 0) for key in failures) else "pass"


def replay_run(options: ReplayOptions) -> dict[str, Any]:
    """Validate a completed run, then atomically publish a research replay.

    Input is expected in nondecreasing step order, as emitted by the staged
    simulator. Complete input tables are never sorted or loaded into memory.
    """
    output = Path(options.output_directory).resolve()
    source = Path(options.run_directory).resolve()
    if output.exists():
        raise ValueError("output_directory already exists; choose a new directory")
    if output.is_relative_to(source):
        raise ValueError("replay output must be outside the immutable source run")
    preflight_started = time.perf_counter()
    metadata, signal_path, signals = _metadata(options)
    preflight_seconds = time.perf_counter() - preflight_started
    assembler = _assembler(metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    row_count = written = invalid = 0
    previous_step = -1
    with tempfile.TemporaryDirectory(prefix=".replay-", dir=output.parent) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        atomic_write_json(staging / "metadata.json", metadata)
        with (
            (staging / "observations.jsonl").open("w", encoding="utf-8") as observations,
            (staging / "frames.jsonl").open("w", encoding="utf-8") as frames,
        ):
            for row in _rows(signal_path, signals, SIGNAL_COLUMNS, batch_size=options.batch_size):
                row_count += 1
                step = _integer(row["step"], "step")
                if step < previous_step or step >= metadata["expected_steps"]:
                    raise ValueError("source rows must have nondecreasing steps within run bounds")
                previous_step = step
                try:
                    observation = observation_from_signal(
                        row,
                        run_id=metadata["run_id"],
                        dt=metadata["dt"],
                        source_id=metadata["source_id"],
                        frame_id=metadata["frame_id"],
                        availability_basis=options.availability_basis,
                    )
                except ObservationError as exc:
                    invalid += 1
                    _write_line(
                        frames,
                        {
                            "type": "observation_rejected",
                            "step": step,
                            "reason": "invalid_observation",
                            "detail": str(exc),
                        },
                    )
                    continue
                envelope = {
                    "step": step,
                    "received_time": observation.available_time,
                    "observation": observation.to_dict(),
                }
                _write_line(observations, envelope)
                written += 1
                for event in assembler.push(
                    observation, step=step, received_time=observation.available_time
                ):
                    _write_line(frames, event)
            for event in assembler.finish():
                _write_line(frames, event)
        summary = assembler.summary()
        report = {
            "replay_version": REPLAY_VERSION,
            "status": _status(summary, invalid),
            "counters": summary,
            "source_rows": row_count,
            "observations_written": written,
            "adapter_rejections": invalid,
            "processing_seconds": time.perf_counter() - started,
            "source_preflight_seconds": preflight_seconds,
            "processing_clock": "monotonic_wall_clock",
            "measured_delivery_latency": False,
            "availability_basis": options.availability_basis,
            "output_sha256": {
                name: sha256_file(staging / name)
                for name in ("metadata.json", "observations.jsonl", "frames.jsonl")
            },
        }
        if row_count != int(signals["row_count"]):
            raise ValueError("source row count disagrees with artifact manifest")
        if sha256_file(signal_path) != metadata["source_sha256"]:
            raise ValueError("source changed while replaying")
        atomic_write_json(staging / "replay_report.json", report)
        if report["status"] == "pass":
            atomic_write_json(staging / "_SUCCESS", {"replay_version": REPLAY_VERSION})
        staging.rename(output)
    return report


def validate_observation_file(path: str | Path, *, metadata_path: str | Path) -> dict[str, Any]:
    """Check canonical JSONL envelopes and causal frame admission without analytics.

    Diagnostics are capped; counters cover the full input. This validates the
    presented stream, not a cryptographic claim that it is the original replay.
    """
    diagnostics: list[dict[str, Any]] = []
    errors = records = 0
    try:
        metadata = _json_object(Path(metadata_path))
        if metadata.get("observation_contract_version") != REPLAY_VERSION:
            raise ValueError("unsupported observation metadata version")
        assembler = _assembler(metadata)
        with Path(path).open(encoding="utf-8") as source:
            while line := source.readline(MAX_RECORD_BYTES + 1):
                records += 1
                if len(line.encode("utf-8")) > MAX_RECORD_BYTES:
                    raise ValueError("observation record exceeds byte limit")
                try:
                    envelope = json.loads(line, object_pairs_hook=_unique_json_keys)
                    if not isinstance(envelope, dict) or set(envelope) != {
                        "step",
                        "received_time",
                        "observation",
                    }:
                        raise ValueError(
                            "envelope must contain only step, received_time, observation"
                        )
                    observation = Observation.from_dict(envelope["observation"])
                    step = _integer(envelope["step"], "step")
                    if isinstance(envelope["step"], str):
                        raise ValueError("canonical step must be an integer, not a string")
                    expected_id = json.dumps(
                        [metadata["stream_id"], step, observation.track_id, "pre_state"],
                        separators=(",", ":"),
                    )
                    if observation.sample_id != expected_id:
                        raise ValueError("sample_id disagrees with its stream, step, and track")
                    for field in ("frame_id", "clock_id", "time_unit", "length_unit", "source_id"):
                        if getattr(observation, field) != metadata[field]:
                            raise ValueError(f"observation {field} disagrees with stream metadata")
                    events = assembler.push(
                        observation, step=envelope["step"], received_time=envelope["received_time"]
                    )
                    for event in events:
                        if (
                            event.get("type")
                            in ("observation_rejected", "frame_skipped", "history_reset")
                            and len(diagnostics) < 20
                        ):
                            diagnostics.append(event)
                except (ValueError, TypeError, KeyError) as exc:
                    errors += 1
                    if len(diagnostics) < 20:
                        diagnostics.append({"record": records, "reason": str(exc)})
        for event in assembler.finish():
            if event.get("type") == "frame_skipped" and len(diagnostics) < 20:
                diagnostics.append(event)
        summary = assembler.summary()
        return {
            "status": _status(summary, errors or int(records == 0)),
            "records": records,
            "invalid_records": errors,
            "counters": summary,
            "diagnostics": diagnostics,
        }
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {
            "status": "fail",
            "records": records,
            "invalid_records": errors + 1,
            "counters": {},
            "diagnostics": [*diagnostics, {"reason": str(exc)}],
        }
