from __future__ import annotations

import json
from pathlib import Path

from drone_swarm.contracts import (
    AGENT_COLUMNS,
    AGENT_TEMPLATE,
    DATASET_CONTRACT_VERSION,
    EVENT_COLUMNS,
    EVENT_TEMPLATE,
    MANIFEST_COLUMNS,
    MANIFEST_TEMPLATE,
    SWARM_TICK_COLUMNS,
    SWARM_TICK_TEMPLATE,
    TABLE_GLOBAL_PRIMARY_KEYS,
    build_run_artifact_manifest,
    inspect_tabular_file,
    preferred_resource_file,
)
from drone_swarm.io import write_records_csv
from drone_swarm.schema_resources import load_schema, write_schema


def test_csv_inspection_and_artifact_manifest(tmp_path: Path) -> None:
    write_records_csv(
        [{"step": 0, "agent_id": 1, "payload": json.dumps({"a": 1})}],
        tmp_path / "agent_signals.csv",
    )
    metadata = inspect_tabular_file(tmp_path / "agent_signals.csv")
    assert metadata["row_count"] == 1
    assert metadata["column_count"] == 3
    assert len(metadata["sha256"]) == 64

    manifest = build_run_artifact_manifest(
        tmp_path,
        run_id="run",
        scenario_id="scenario",
    )
    assert manifest["contract_version"] == DATASET_CONTRACT_VERSION
    selected = preferred_resource_file(manifest, "agent_signals")
    assert selected is not None
    assert selected["path"] == "agent_signals.csv"


def test_bundled_json_schemas_load_and_can_be_exported(tmp_path: Path) -> None:
    scenario = load_schema("scenario")
    assert scenario["$schema"].endswith("2020-12/schema")
    output = write_schema("dataset-catalog", tmp_path / "catalog.schema.json")
    assert output.is_file()
    assert json.loads(output.read_text())["title"] == "Drone Swarm Dataset Catalog"


def test_fixed_table_templates_cover_declared_columns_and_global_keys() -> None:
    assert tuple(EVENT_TEMPLATE) == EVENT_COLUMNS
    assert tuple(SWARM_TICK_TEMPLATE) == SWARM_TICK_COLUMNS
    assert tuple(AGENT_TEMPLATE) == AGENT_COLUMNS
    assert tuple(MANIFEST_TEMPLATE) == MANIFEST_COLUMNS
    assert TABLE_GLOBAL_PRIMARY_KEYS["transitions"] == ("run_id", "step", "agent_id")
    assert TABLE_GLOBAL_PRIMARY_KEYS["events"] == ("run_id", "event_id")
