"""Versioned data contracts and artifact-inspection helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from drone_swarm.io import sha256_file

DATASET_CONTRACT_VERSION = "1.0.0"
TRANSITIONS_CONTRACT_VERSION = "1.0.0"
AGENT_SIGNALS_CONTRACT_VERSION = "1.0.0"
SWARM_TICKS_CONTRACT_VERSION = "1.0.0"
AGENTS_CONTRACT_VERSION = "1.0.0"
EVENTS_CONTRACT_VERSION = "1.0.0"

TABLE_CONTRACT_VERSIONS = {
    "transitions": TRANSITIONS_CONTRACT_VERSION,
    "agent_signals": AGENT_SIGNALS_CONTRACT_VERSION,
    "swarm_ticks": SWARM_TICKS_CONTRACT_VERSION,
    "agents": AGENTS_CONTRACT_VERSION,
    "events": EVENTS_CONTRACT_VERSION,
}

TABLE_STEMS = ("transitions", "agent_signals", "swarm_ticks", "agents", "events")
MANIFEST_COLUMNS = (
    "contract_version",
    "run_id",
    "scenario_id",
    "analysis_split",
    "paired_control_scenario",
    "paired_treatment_scenario",
    "config_hash",
    "simulation_hash",
    "repository_commit",
    "base_seed",
    "initialization_seed",
    "policy_seed",
    "perturbation_seed",
    "n_drones",
    "steps",
    "dt",
    "activation",
    "event_types",
    "event_count",
    "transition_rows",
    "agent_signal_rows",
    "metrics_rows",
    "formats",
    "transitions_file",
    "agent_signals_file",
    "swarm_ticks_file",
    "agents_file",
    "events_file",
    "started_at_utc",
    "ended_at_utc",
    "runtime_seconds",
    "status",
    "run_path",
    "transitions_path",
    "agent_signals_path",
    "swarm_ticks_path",
    "agents_path",
    "events_path",
)
TABLE_PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "transitions": ("step", "agent_id"),
    "agent_signals": ("step", "agent_id"),
    "swarm_ticks": ("tick",),
    "agents": ("agent_id",),
    "events": ("event_id",),
}
TABLE_GLOBAL_PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "transitions": ("run_id", "step", "agent_id"),
    "agent_signals": ("run_id", "step", "agent_id"),
    "swarm_ticks": ("run_id", "tick"),
    "agents": ("run_id", "agent_id"),
    "events": ("run_id", "event_id"),
}
TABLE_SEMANTICS: dict[str, str] = {
    "transitions": "Canonical controller- and plant-facing state-action-outcome records.",
    "agent_signals": "Compact analysis projection of canonical transitions.",
    "swarm_ticks": "Conventional swarm and realized interaction-graph diagnostics.",
    "agents": "Stable agent registry with initial and final planted metadata.",
    "events": "Configured intervention schedule and resolved target population.",
}


EVENT_COLUMNS = (
    "run_id",
    "scenario_id",
    "base_seed",
    "event_id",
    "event_type",
    "intent",
    "apply_stage",
    "start_step",
    "end_step",
    "schedule",
    "ramp_steps",
    "period",
    "duty_cycle",
    "target_agent_indices",
    "severity_json",
    "restore",
    "description",
)
EVENT_TEMPLATE: dict[str, Any] = {
    "run_id": "run-template",
    "scenario_id": "scenario-template",
    "base_seed": 0,
    "event_id": "event-template",
    "event_type": "wind",
    "intent": "fault",
    "apply_stage": "physics",
    "start_step": 0,
    "end_step": 1,
    "schedule": "step",
    "ramp_steps": 0,
    "period": 1,
    "duty_cycle": 1.0,
    "target_agent_indices": "[]",
    "severity_json": "{}",
    "restore": False,
    "description": "template",
}

SWARM_TICK_COLUMNS = (
    "run_id",
    "scenario_id",
    "base_seed",
    "tick",
    "phase",
    "n_drones",
    "transition_count",
    "mean_speed",
    "min_pairwise_distance",
    "collision_count",
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "polarization",
    "radius_of_gyration",
    "position_eigenvalue_1",
    "position_eigenvalue_2",
    "position_eigenvalue_3",
    "position_anisotropy",
    "interaction_mean_degree",
    "interaction_component_count",
    "interaction_largest_component_fraction",
    "interaction_algebraic_connectivity",
)
SWARM_TICK_TEMPLATE: dict[str, Any] = {
    "run_id": "run-template",
    "scenario_id": "scenario-template",
    "base_seed": 0,
    "tick": 0,
    "phase": "template",
    "n_drones": 0,
    "transition_count": 0,
    "mean_speed": 0.0,
    "min_pairwise_distance": 0.0,
    "collision_count": 0,
    "centroid_x": 0.0,
    "centroid_y": 0.0,
    "centroid_z": 0.0,
    "polarization": 0.0,
    "radius_of_gyration": 0.0,
    "position_eigenvalue_1": 0.0,
    "position_eigenvalue_2": 0.0,
    "position_eigenvalue_3": 0.0,
    "position_anisotropy": 0.0,
    "interaction_mean_degree": 0.0,
    "interaction_component_count": 0,
    "interaction_largest_component_fraction": 0.0,
    "interaction_algebraic_connectivity": 0.0,
}

AGENT_COLUMNS = (
    "run_id",
    "scenario_id",
    "base_seed",
    "agent_id",
    "agent_index",
    "initial_coalition_id",
    "initial_role",
    "initial_formation_id",
    "initial_target_id",
    "initial_target_x",
    "initial_target_y",
    "initial_target_z",
    "initial_restrict_interactions_to_coalition",
    "final_coalition_id",
    "final_role",
    "final_formation_id",
    "final_target_id",
    "final_target_x",
    "final_target_y",
    "final_target_z",
    "final_restrict_interactions_to_coalition",
)
AGENT_TEMPLATE: dict[str, Any] = {
    "run_id": "run-template",
    "scenario_id": "scenario-template",
    "base_seed": 0,
    "agent_id": 0,
    "agent_index": 0,
    "initial_coalition_id": "all",
    "initial_role": "member",
    "initial_formation_id": "cloud",
    "initial_target_id": "target",
    "initial_target_x": 0.0,
    "initial_target_y": 0.0,
    "initial_target_z": 0.0,
    "initial_restrict_interactions_to_coalition": False,
    "final_coalition_id": "all",
    "final_role": "member",
    "final_formation_id": "cloud",
    "final_target_id": "target",
    "final_target_x": 0.0,
    "final_target_y": 0.0,
    "final_target_z": 0.0,
    "final_restrict_interactions_to_coalition": False,
}

MANIFEST_TEMPLATE: dict[str, Any] = {
    "contract_version": DATASET_CONTRACT_VERSION,
    "run_id": "run-template",
    "scenario_id": "scenario-template",
    "analysis_split": "pilot",
    "paired_control_scenario": "control-template",
    "paired_treatment_scenario": "treatment-template",
    "config_hash": "0" * 64,
    "simulation_hash": "0" * 64,
    "repository_commit": "unknown",
    "base_seed": 0,
    "initialization_seed": 0,
    "policy_seed": 0,
    "perturbation_seed": 0,
    "n_drones": 0,
    "steps": 0,
    "dt": 0.0,
    "activation": "staged",
    "event_types": "[]",
    "event_count": 0,
    "transition_rows": 0,
    "agent_signal_rows": 0,
    "metrics_rows": 0,
    "formats": "[]",
    "transitions_file": "transitions.parquet",
    "agent_signals_file": "agent_signals.parquet",
    "swarm_ticks_file": "swarm_ticks.parquet",
    "agents_file": "agents.parquet",
    "events_file": "events.parquet",
    "started_at_utc": "1970-01-01T00:00:00+00:00",
    "ended_at_utc": "1970-01-01T00:00:00+00:00",
    "runtime_seconds": 0.0,
    "status": "success",
    "run_path": "raw/run-template",
    "transitions_path": "raw/run-template/transitions.parquet",
    "agent_signals_path": "raw/run-template/agent_signals.parquet",
    "swarm_ticks_path": "raw/run-template/swarm_ticks.parquet",
    "agents_path": "raw/run-template/agents.parquet",
    "events_path": "raw/run-template/events.parquet",
}


TRANSITION_REQUIRED_COLUMNS = (
    "run_id",
    "episode_id",
    "step",
    "agent_id",
    "phase",
    "coalition_truth",
    "s_position_x",
    "s_position_y",
    "s_position_z",
    "s_velocity_x",
    "s_velocity_y",
    "s_velocity_z",
    "a_acceleration_x",
    "a_acceleration_y",
    "a_acceleration_z",
    "sp_position_x",
    "sp_position_y",
    "sp_position_z",
    "true_s_position_x",
    "applied_a_acceleration_x",
    "true_sp_position_x",
)

AGENT_SIGNALS_REQUIRED_COLUMNS = (
    "contract_version",
    "run_id",
    "scenario_id",
    "base_seed",
    "step",
    "agent_id",
    "agent_index",
    "phase",
    "coalition_truth",
    "observed_state_position_x",
    "observed_state_velocity_x",
    "commanded_action_acceleration_x",
    "observed_next_state_position_x",
    "true_state_position_x",
    "applied_action_acceleration_x",
    "true_next_state_position_x",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def schema_fingerprint(columns: Sequence[Mapping[str, Any]]) -> str:
    """Hash an ordered list of column descriptions."""

    return hashlib.sha256(canonical_json(list(columns)).encode("utf-8")).hexdigest()


def _count_text_rows(path: Path, *, header: bool) -> int:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as file_obj:
            count = sum(1 for row in csv.reader(file_obj) if row)
    else:
        with path.open(encoding="utf-8") as file_obj:
            count = sum(1 for line in file_obj if line.strip())
    return max(0, count - (1 if header and count else 0))


def _csv_columns(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        reader = csv.reader(file_obj)
        header = next(reader, [])
    return [
        {"name": str(name), "type": "unknown/csv", "nullable": True, "position": index}
        for index, name in enumerate(header)
    ]


def _jsonl_columns(path: Path) -> list[dict[str, Any]]:
    first: Mapping[str, Any] = {}
    with path.open(encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, Mapping):
                    first = value
                break
    return [
        {
            "name": str(name),
            "type": type(value).__name__,
            "nullable": value is None,
            "position": index,
        }
        for index, (name, value) in enumerate(first.items())
    ]


def inspect_tabular_file(path: str | Path) -> dict[str, Any]:
    """Return portable metadata for one Parquet, CSV, or JSONL artifact."""

    table_path = Path(path)
    suffix = table_path.suffix.lower()
    columns: list[dict[str, Any]]
    row_count: int
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Inspecting Parquet requires PyArrow") from exc
        parquet = pq.ParquetFile(table_path)
        row_count = int(parquet.metadata.num_rows)
        arrow_schema = parquet.schema_arrow
        columns = [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": bool(field.nullable),
                "position": index,
            }
            for index, field in enumerate(arrow_schema)
        ]
        format_name = "parquet"
        media_type = "application/vnd.apache.parquet"
    elif suffix == ".csv":
        row_count = _count_text_rows(table_path, header=True)
        columns = _csv_columns(table_path)
        format_name = "csv"
        media_type = "text/csv"
    elif suffix == ".jsonl":
        row_count = _count_text_rows(table_path, header=False)
        columns = _jsonl_columns(table_path)
        format_name = "jsonl"
        media_type = "application/x-ndjson"
    else:
        raise ValueError(f"unsupported tabular artifact: {table_path}")

    return {
        "path": table_path.name,
        "format": format_name,
        "media_type": media_type,
        "size_bytes": table_path.stat().st_size,
        "sha256": sha256_file(table_path),
        "row_count": row_count,
        "column_count": len(columns),
        "schema_fingerprint": schema_fingerprint(columns),
        "columns": columns,
    }


def discover_table_files(directory: str | Path, stem: str) -> list[Path]:
    root = Path(directory)
    return [
        path
        for suffix in (".parquet", ".csv", ".jsonl")
        if (path := root / f"{stem}{suffix}").is_file()
    ]


def build_run_artifact_manifest(
    directory: str | Path,
    *,
    run_id: str,
    scenario_id: str,
) -> dict[str, Any]:
    """Describe all tabular resources in a completed or partial run directory."""

    root = Path(directory)
    resources: list[dict[str, Any]] = []
    for stem in TABLE_STEMS:
        files = [inspect_tabular_file(path) for path in discover_table_files(root, stem)]
        if not files:
            continue
        resources.append(
            {
                "name": stem,
                "contract_version": TABLE_CONTRACT_VERSIONS[stem],
                "description": TABLE_SEMANTICS[stem],
                "primary_key": list(TABLE_PRIMARY_KEYS[stem]),
                "global_primary_key": list(TABLE_GLOBAL_PRIMARY_KEYS[stem]),
                "files": files,
            }
        )

    return {
        "contract_version": DATASET_CONTRACT_VERSION,
        "run_id": run_id,
        "scenario_id": scenario_id,
        "resources": resources,
    }


def resource_by_name(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    resources = manifest.get("resources", ())
    if not isinstance(resources, Iterable):
        return None
    for resource in resources:
        if isinstance(resource, Mapping) and resource.get("name") == name:
            return resource
    return None


def preferred_resource_file(
    manifest: Mapping[str, Any],
    name: str,
    *,
    preference: Sequence[str] = ("parquet", "csv", "jsonl"),
) -> Mapping[str, Any] | None:
    resource = resource_by_name(manifest, name)
    if resource is None:
        return None
    files = resource.get("files", ())
    if not isinstance(files, Sequence):
        return None
    for format_name in preference:
        for file_value in files:
            if isinstance(file_value, Mapping) and file_value.get("format") == format_name:
                return file_value
    return None
