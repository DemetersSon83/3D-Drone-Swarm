from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from drone_swarm.dataset import build_dataset_catalog
from drone_swarm.validation import validate_run_directory


def test_run_validation_and_dataset_catalog(
    make_fake_run: Callable[..., Path],
) -> None:
    root = make_fake_run(scenario_id="nominal", base_seed=1)
    run_dir = next((root / "raw").iterdir())
    report = validate_run_directory(run_dir, level="full")
    assert report["status"] == "pass"

    summary = build_dataset_catalog(root, strict=True, no_parquet=True, validation_level="full")
    assert summary["successful_runs"] == 1
    assert summary["transition_rows"] == 4
    assert summary["quality_status"] == "pass"
    for relative in (
        "_SUCCESS",
        "manifest.csv",
        "dataset_catalog.json",
        "datapackage.json",
        "schema_registry.json",
        "dataset_summary.json",
        "quality_report.json",
        "DATASET_CARD.md",
        "checksums.sha256",
        "sql/duckdb_views.sql",
        "schemas/scenario.schema.json",
        "schemas/artifact-manifest.schema.json",
        "schemas/dataset-catalog.schema.json",
    ):
        assert (root / relative).is_file(), relative

    catalog = json.loads((root / "dataset_catalog.json").read_text())
    assert catalog["tables"]["agent_signals"]["preferred_format"] == "csv"
    assert catalog["tables"]["agent_signals"]["global_primary_key"] == [
        "run_id",
        "step",
        "agent_id",
    ]
    assert catalog["run_count"] == 1


def test_strict_catalog_fails_missing_declared_control(
    make_fake_run: Callable[..., Path],
) -> None:
    root = make_fake_run(
        scenario_id="treatment",
        base_seed=5,
        paired_control_scenario="control",
    )
    summary = build_dataset_catalog(root, strict=True, no_parquet=True)
    assert summary["quality_status"] == "fail"
    quality = json.loads((root / "quality_report.json").read_text())
    assert quality["pairing_warnings"]
