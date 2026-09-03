"""Run- and dataset-level validation for generated swarm artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from drone_swarm.contracts import (
    AGENT_SIGNALS_REQUIRED_COLUMNS,
    DATASET_CONTRACT_VERSION,
    TRANSITION_REQUIRED_COLUMNS,
    preferred_resource_file,
    resource_by_name,
)
from drone_swarm.io import sha256_file

ValidationLevel = Literal["quick", "standard", "full"]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _add(
    checks: list[dict[str, Any]],
    *,
    check: str,
    passed: bool,
    severity: str = "error",
    detail: str = "",
) -> None:
    checks.append(
        {
            "check": check,
            "passed": bool(passed),
            "severity": severity,
            "detail": detail,
        }
    )


def _column_names(file_metadata: Mapping[str, Any]) -> set[str]:
    columns = file_metadata.get("columns", ())
    if not isinstance(columns, Sequence):
        return set()
    return {
        str(column.get("name"))
        for column in columns
        if isinstance(column, Mapping) and column.get("name") is not None
    }


def _read_columns(path: Path, columns: Sequence[str]) -> Any:
    import pandas as pd

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=list(columns))
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=list(columns))
    raise ValueError(f"full validation does not support {path.suffix}: {path}")


def _validate_transition_keys(
    path: Path,
    *,
    expected_steps: int,
    expected_agents: int,
    identity_column: str,
    expected_run_id: str,
) -> tuple[bool, str]:
    try:
        frame = _read_columns(path, ("step", "agent_id", identity_column))
    except (ImportError, RuntimeError, ValueError) as exc:
        return False, f"full key validation was skipped: {exc}"

    expected_rows = expected_steps * expected_agents
    if frame.empty:
        passed = expected_rows == 0
        return passed, f"rows=0/{expected_rows}"

    observed_steps = frame["step"].nunique(dropna=False)
    observed_agents = frame["agent_id"].nunique(dropna=False)
    unique_keys = frame[["step", "agent_id"]].drop_duplicates().shape[0]
    identities = {str(value) for value in frame[identity_column].dropna().unique()}
    passed = (
        len(frame) == expected_rows
        and int(frame["step"].min()) == 0
        and int(frame["step"].max()) == expected_steps - 1
        and observed_steps == expected_steps
        and observed_agents == expected_agents
        and unique_keys == expected_rows
        and identities == {expected_run_id}
    )
    detail = (
        f"rows={len(frame)}/{expected_rows}, steps={observed_steps}/{expected_steps}, "
        f"agents={observed_agents}/{expected_agents}, unique_keys={unique_keys}/{expected_rows}, "
        f"identities={sorted(identities)}"
    )
    return passed, detail


def validate_run_directory(
    directory: str | Path,
    *,
    level: ValidationLevel = "standard",
    require_success: bool = True,
    verify_checksums: bool = True,
) -> dict[str, Any]:
    """Validate one run directory.

    ``quick`` checks metadata and declared cardinalities. ``standard`` adds
    required-column and checksum checks. ``full`` reads key columns to verify
    uniqueness, coverage, and run identity.
    """

    if level not in {"quick", "standard", "full"}:
        raise ValueError(f"unsupported validation level: {level}")

    root = Path(directory)
    checks: list[dict[str, Any]] = []
    required_metadata = (
        "config.json",
        "provenance.json",
        "manifest_row.json",
        "artifact_manifest.json",
    )
    for name in required_metadata:
        _add(
            checks,
            check=f"metadata:{name}",
            passed=(root / name).is_file(),
            detail=str(root / name),
        )
    if require_success:
        _add(
            checks,
            check="completion_marker",
            passed=(root / "_SUCCESS").is_file(),
            detail=str(root / "_SUCCESS"),
        )

    try:
        config = _load_json(root / "config.json")
        manifest_row = _load_json(root / "manifest_row.json")
        artifacts = _load_json(root / "artifact_manifest.json")
        provenance = _load_json(root / "provenance.json")
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _add(checks, check="metadata_parse", passed=False, detail=str(exc))
        return _finish_report(root, checks, level)

    run_id = str(manifest_row.get("run_id", ""))
    declared_ids = {
        "config": str(config.get("run_id", "")),
        "manifest": run_id,
        "artifacts": str(artifacts.get("run_id", "")),
        "provenance": str(provenance.get("run_id", "")),
    }
    _add(
        checks,
        check="run_id_consistency",
        passed=bool(run_id) and set(declared_ids.values()) == {run_id},
        detail=json.dumps(declared_ids, sort_keys=True),
    )
    if not root.name.startswith("."):
        _add(
            checks,
            check="run_directory_name",
            passed=root.name == run_id,
            severity="warning",
            detail=f"directory={root.name}, run_id={run_id}",
        )
    contracts = {
        str(config.get("contract_version", "")),
        str(manifest_row.get("contract_version", "")),
        str(artifacts.get("contract_version", "")),
        str(provenance.get("contract_version", "")),
    }
    _add(
        checks,
        check="contract_version_consistency",
        passed=contracts == {DATASET_CONTRACT_VERSION},
        detail=f"observed={sorted(contracts)}, expected={DATASET_CONTRACT_VERSION}",
    )

    scenario = config.get("scenario", {})
    if not isinstance(scenario, Mapping):
        scenario = {}
    run = scenario.get("run", {})
    model = scenario.get("model", {})
    if not isinstance(run, Mapping):
        run = {}
    if not isinstance(model, Mapping):
        model = {}
    expected_steps = int(run.get("steps", manifest_row.get("steps", 0)))
    expected_agents = int(model.get("n_drones", manifest_row.get("n_drones", 0)))
    expected_rows = expected_steps * expected_agents
    metrics_stride = int(run.get("metrics_stride", 1))
    expected_metric_rows = math.ceil(expected_steps / metrics_stride) if expected_steps else 0
    expected_events = int(manifest_row.get("event_count", 0))

    transitions = resource_by_name(artifacts, "transitions")
    _add(
        checks,
        check="resource:transitions",
        passed=transitions is not None,
        detail="canonical transition table must exist",
    )
    transition_file = preferred_resource_file(artifacts, "transitions")
    if transition_file is not None:
        rows = int(transition_file.get("row_count", -1))
        _add(
            checks,
            check="transition_row_count",
            passed=rows == expected_rows,
            detail=f"observed={rows}, expected={expected_rows}",
        )
        if level != "quick":
            names = _column_names(transition_file)
            missing = sorted(set(TRANSITION_REQUIRED_COLUMNS).difference(names))
            _add(
                checks,
                check="transition_required_columns",
                passed=not missing,
                detail="missing=" + json.dumps(missing),
            )

    signals_file = preferred_resource_file(artifacts, "agent_signals")
    if signals_file is not None:
        rows = int(signals_file.get("row_count", -1))
        _add(
            checks,
            check="agent_signals_row_count",
            passed=rows == expected_rows,
            detail=f"observed={rows}, expected={expected_rows}",
        )
        if level != "quick":
            names = _column_names(signals_file)
            missing = sorted(set(AGENT_SIGNALS_REQUIRED_COLUMNS).difference(names))
            _add(
                checks,
                check="agent_signals_required_columns",
                passed=not missing,
                detail="missing=" + json.dumps(missing),
            )
    else:
        _add(
            checks,
            check="resource:agent_signals",
            passed=False,
            severity="warning",
            detail="compact projection was not requested",
        )

    expected_resources = {
        "agents": expected_agents,
        "events": expected_events,
        "swarm_ticks": expected_metric_rows,
    }
    for resource_name, expected_count in expected_resources.items():
        selected = preferred_resource_file(artifacts, resource_name)
        _add(
            checks,
            check=f"resource:{resource_name}",
            passed=selected is not None,
            detail=f"expected rows={expected_count}",
        )
        if selected is not None:
            observed_count = int(selected.get("row_count", -1))
            _add(
                checks,
                check=f"{resource_name}_row_count",
                passed=observed_count == expected_count,
                detail=f"observed={observed_count}, expected={expected_count}",
            )

    if verify_checksums and level != "quick":
        for resource in artifacts.get("resources", ()):
            if not isinstance(resource, Mapping):
                continue
            for file_value in resource.get("files", ()):
                if not isinstance(file_value, Mapping):
                    continue
                relative = str(file_value.get("path", ""))
                path = root / relative
                expected_sha = str(file_value.get("sha256", ""))
                actual_sha = sha256_file(path) if path.is_file() else "missing"
                _add(
                    checks,
                    check=f"checksum:{relative}",
                    passed=actual_sha == expected_sha,
                    detail=f"expected={expected_sha}, actual={actual_sha}",
                )

    if level == "full" and transition_file is not None:
        key_ok, detail = _validate_transition_keys(
            root / str(transition_file["path"]),
            expected_steps=expected_steps,
            expected_agents=expected_agents,
            identity_column="run_id",
            expected_run_id=run_id,
        )
        _add(
            checks,
            check="transition_primary_key_and_identity",
            passed=key_ok,
            severity="warning" if "skipped" in detail else "error",
            detail=detail,
        )
    if level == "full" and signals_file is not None:
        key_ok, detail = _validate_transition_keys(
            root / str(signals_file["path"]),
            expected_steps=expected_steps,
            expected_agents=expected_agents,
            identity_column="run_id",
            expected_run_id=run_id,
        )
        _add(
            checks,
            check="agent_signals_primary_key_and_identity",
            passed=key_ok,
            severity="warning" if "skipped" in detail else "error",
            detail=detail,
        )

    return _finish_report(root, checks, level)


def _finish_report(
    root: Path,
    checks: list[dict[str, Any]],
    level: ValidationLevel,
) -> dict[str, Any]:
    errors = [item for item in checks if not item["passed"] and item.get("severity") == "error"]
    warnings = [item for item in checks if not item["passed"] and item.get("severity") == "warning"]
    return {
        "validation_version": "1.0.0",
        "path": str(root),
        "level": level,
        "status": "fail" if errors else "warning" if warnings else "pass",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "checks": checks,
    }
