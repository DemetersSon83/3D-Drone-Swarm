from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from drone_swarm.cli import main
from drone_swarm.contracts import build_run_artifact_manifest
from drone_swarm.io import atomic_write_json, write_records_csv
from drone_swarm.replay import ReplayOptions, replay_run, validate_observation_file


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _signals(run: Path) -> list[dict[str, Any]]:
    with (run / "agent_signals.csv").open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def _refresh_artifacts(run: Path) -> None:
    manifest = _json(run / "manifest_row.json")
    artifacts = build_run_artifact_manifest(
        run,
        run_id=manifest["run_id"],
        scenario_id=manifest["scenario_id"],
    )
    atomic_write_json(run / "artifact_manifest.json", artifacts)
    provenance = _json(run / "provenance.json")
    provenance["resources"] = artifacts["resources"]
    atomic_write_json(run / "provenance.json", provenance)


@pytest.fixture
def source_run(make_fake_run: Callable[..., Path]) -> Path:
    root = make_fake_run()
    return next((root / "raw").iterdir())


def test_replay_preserves_observed_pre_state_and_explicit_clock(
    source_run: Path, tmp_path: Path
) -> None:
    output = tmp_path / "replayed"
    report = replay_run(ReplayOptions(source_run, output, batch_size=1))

    assert report["status"] == "pass"
    assert (output / "_SUCCESS").is_file()
    assert _json(output / "replay_report.json") == report
    envelopes = _jsonl(output / "observations.jsonl")
    assert len(envelopes) == 4
    observations = [envelope["observation"] for envelope in envelopes]
    assert [item["event_time"] for item in observations] == [0.0, 0.0, 0.25, 0.25]
    assert [item["available_time"] for item in observations] == [0.25, 0.25, 0.5, 0.5]
    assert [item["position"][0] for item in observations] == [0.0, 1.0, 1.0, 2.0]
    assert len({item["sample_id"] for item in observations}) == 4
    assert {item["track_id"] for item in observations} == {"1", "2"}
    for envelope in envelopes:
        observation = envelope["observation"]
        assert envelope["received_time"] >= observation["available_time"]
        assert observation["availability_basis"] == "completed_transition_lower_bound"
        assert observation["sample_phase"] == "pre_state"
        assert observation["time_unit"] == "simulation_time"
        assert observation["length_unit"] == "simulation_length"
        assert observation["position_covariance"] is None
        assert observation["velocity_covariance"] is None
        assert not observation["is_imputed"]
        assert not observation["is_extrapolated"]
    assert len(_jsonl(output / "frames.jsonl")) == 2
    validated = validate_observation_file(
        output / "observations.jsonl", metadata_path=output / "metadata.json"
    )
    assert validated["status"] == "pass"


def test_idealized_replay_changes_availability_without_changing_measurements(
    source_run: Path, tmp_path: Path
) -> None:
    completed = tmp_path / "completed"
    idealized = tmp_path / "idealized"
    replay_run(ReplayOptions(source_run, completed))
    replay_run(
        ReplayOptions(source_run, idealized, availability_basis="idealized_pre_state_replay")
    )
    default_rows = _jsonl(completed / "observations.jsonl")
    ideal_rows = _jsonl(idealized / "observations.jsonl")
    assert len(default_rows) == len(ideal_rows) == 4
    for completed_row, ideal_row in zip(default_rows, ideal_rows, strict=True):
        before = completed_row["observation"]
        after = ideal_row["observation"]
        assert after["available_time"] == after["event_time"]
        assert after["availability_basis"] == "idealized_pre_state_replay"
        for key in ("position", "velocity", "event_time", "track_id", "sample_id"):
            assert before[key] == after[key]
        assert completed_row["received_time"] == ideal_row["received_time"] + 0.25


def test_future_states_and_evaluator_labels_cannot_change_payload(
    source_run: Path, tmp_path: Path
) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    replay_run(ReplayOptions(source_run, before))
    rows = _signals(source_run)
    for row in rows:
        for key in tuple(row):
            if key.startswith(("true_", "observed_next_state_", "applied_action_")):
                row[key] = "DO_NOT_USE_THIS_FUTURE_OR_TRUTH_FIELD"
        for key in (
            "phase",
            "active_event_ids",
            "agent_affected",
            "coalition_truth",
            "role_truth",
            "formation_truth",
            "target_id",
            "reward",
        ):
            row[key] = "DO_NOT_USE_THIS_EVALUATOR_LABEL"
    write_records_csv(rows, source_run / "agent_signals.csv")
    _refresh_artifacts(source_run)
    replay_run(ReplayOptions(source_run, after))
    assert _jsonl(before / "observations.jsonl") == _jsonl(after / "observations.jsonl")
    assert "DO_NOT_USE" not in (after / "observations.jsonl").read_text(encoding="utf-8")


def test_source_checksum_tampering_fails_before_output_commit(
    source_run: Path, tmp_path: Path
) -> None:
    with (source_run / "agent_signals.csv").open("a", encoding="utf-8") as file_obj:
        file_obj.write("\n")
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="(?i)(hash|checksum|size)"):
        replay_run(ReplayOptions(source_run, output))
    assert not output.exists()


@pytest.mark.parametrize("filename", ["config.json", "manifest_row.json", "provenance.json"])
def test_source_metadata_duplicate_keys_fail_before_legacy_validation(
    source_run: Path, tmp_path: Path, monkeypatch: Any, filename: str
) -> None:
    metadata_path = source_run / filename
    original = metadata_path.read_text(encoding="utf-8")
    metadata_path.write_text('{"run_id": "ignored-duplicate", ' + original[1:], encoding="utf-8")

    def unexpected_legacy_validation(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("ambiguous source metadata reached legacy validation")

    monkeypatch.setattr("drone_swarm.replay.validate_run_directory", unexpected_legacy_validation)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="duplicate JSON key: run_id"):
        replay_run(ReplayOptions(source_run, output))
    assert not output.exists()


def test_oversized_source_provenance_fails_before_legacy_validation(
    source_run: Path, tmp_path: Path, monkeypatch: Any
) -> None:
    provenance = _json(source_run / "provenance.json")
    provenance["padding"] = "x" * (4 * 1024 * 1024)
    atomic_write_json(source_run / "provenance.json", provenance)

    def unexpected_legacy_validation(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("oversized source metadata reached legacy validation")

    monkeypatch.setattr("drone_swarm.replay.validate_run_directory", unexpected_legacy_validation)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match=r"metadata exceeds .*provenance\.json"):
        replay_run(ReplayOptions(source_run, output))
    assert not output.exists()


def test_manifest_path_cannot_escape_run_directory(source_run: Path, tmp_path: Path) -> None:
    artifacts = _json(source_run / "artifact_manifest.json")
    signals = next(
        resource for resource in artifacts["resources"] if resource["name"] == "agent_signals"
    )
    signals["files"][0]["path"] = "../outside.csv"
    atomic_write_json(source_run / "artifact_manifest.json", artifacts)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        replay_run(ReplayOptions(source_run, output))
    assert not output.exists()


@pytest.mark.parametrize("activation", ["random", "sequential", ""])
def test_replay_requires_staged_activation(
    source_run: Path, tmp_path: Path, activation: str
) -> None:
    manifest = _json(source_run / "manifest_row.json")
    manifest["activation"] = activation
    atomic_write_json(source_run / "manifest_row.json", manifest)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="(?i)(staged|activation)"):
        replay_run(ReplayOptions(source_run, output))
    assert not output.exists()


@pytest.mark.parametrize("dt", [0.0, -0.25, float("nan")])
def test_replay_requires_positive_finite_source_period(
    source_run: Path, tmp_path: Path, dt: float
) -> None:
    manifest = _json(source_run / "manifest_row.json")
    config = _json(source_run / "config.json")
    manifest["dt"] = dt
    config["scenario"]["model"]["dt"] = dt
    atomic_write_json(source_run / "manifest_row.json", manifest)
    atomic_write_json(source_run / "config.json", config)
    with pytest.raises(ValueError):
        replay_run(ReplayOptions(source_run, tmp_path / "rejected"))


def test_huge_integer_source_period_is_a_clean_validation_error(
    source_run: Path, tmp_path: Path
) -> None:
    config = _json(source_run / "config.json")
    manifest = _json(source_run / "manifest_row.json")
    config["scenario"]["model"]["dt"] = 10**400
    manifest["dt"] = 10**400
    atomic_write_json(source_run / "config.json", config)
    atomic_write_json(source_run / "manifest_row.json", manifest)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="dt must be a finite number"):
        replay_run(ReplayOptions(source_run, output))
    assert not output.exists()


def test_replay_rejects_declared_cohort_over_capacity(source_run: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="(?i)(cap|track|cohort)"):
        replay_run(ReplayOptions(source_run, tmp_path / "rejected", max_tracks=1))


def test_missing_completion_marker_is_not_replayed(source_run: Path, tmp_path: Path) -> None:
    (source_run / "_SUCCESS").unlink()
    with pytest.raises(ValueError):
        replay_run(ReplayOptions(source_run, tmp_path / "rejected"))


def test_failed_source_quality_is_not_replayed(source_run: Path, tmp_path: Path) -> None:
    completion = _json(source_run / "_SUCCESS")
    completion["quality_status"] = "fail"
    atomic_write_json(source_run / "_SUCCESS", completion)
    with pytest.raises(ValueError):
        replay_run(ReplayOptions(source_run, tmp_path / "rejected"))


def test_missing_observed_column_is_a_source_schema_failure(
    source_run: Path, tmp_path: Path
) -> None:
    rows = _signals(source_run)
    for row in rows:
        del row["observed_state_position_z"]
    write_records_csv(rows, source_run / "agent_signals.csv")
    _refresh_artifacts(source_run)
    with pytest.raises(ValueError):
        replay_run(ReplayOptions(source_run, tmp_path / "rejected"))


def test_existing_output_is_never_overwritten(source_run: Path, tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("user content", encoding="utf-8")
    with pytest.raises((ValueError, FileExistsError)):
        replay_run(ReplayOptions(source_run, output))
    assert sentinel.read_text(encoding="utf-8") == "user content"
    assert list(output.iterdir()) == [sentinel]


def test_parquet_and_csv_produce_identical_observations(source_run: Path, tmp_path: Path) -> None:
    pa_csv = pytest.importorskip("pyarrow.csv")
    pq = pytest.importorskip("pyarrow.parquet")
    csv_output = tmp_path / "csv"
    parquet_output = tmp_path / "parquet"
    replay_run(ReplayOptions(source_run, csv_output, batch_size=1))
    pq.write_table(
        pa_csv.read_csv(source_run / "agent_signals.csv"), source_run / "agent_signals.parquet"
    )
    (source_run / "agent_signals.csv").unlink()
    manifest = _json(source_run / "manifest_row.json")
    manifest["agent_signals_file"] = "agent_signals.parquet"
    atomic_write_json(source_run / "manifest_row.json", manifest)
    _refresh_artifacts(source_run)
    replay_run(ReplayOptions(source_run, parquet_output, batch_size=1))
    assert _jsonl(csv_output / "observations.jsonl") == _jsonl(
        parquet_output / "observations.jsonl"
    )


@pytest.mark.parametrize("value", ["nan", "inf", "not-a-number"])
def test_invalid_position_is_rejected_without_a_fabricated_complete_frame(
    source_run: Path, tmp_path: Path, value: str
) -> None:
    rows = _signals(source_run)
    rows[0]["observed_state_position_x"] = value
    write_records_csv(rows, source_run / "agent_signals.csv")
    _refresh_artifacts(source_run)
    output = tmp_path / "rejected-row"
    report = replay_run(ReplayOptions(source_run, output))
    assert report["status"] in {"fail", "degraded"}
    observations = _jsonl(output / "observations.jsonl")
    assert len(observations) == 3
    assert all(item["observation"]["position"] is not None for item in observations)
    assert report["adapter_rejections"] == 1
    assert report["counters"]["frames_ready"] == 1
    assert report["counters"]["frames_skipped"] == 1
    skipped = [
        event for event in _jsonl(output / "frames.jsonl") if event["type"] == "frame_skipped"
    ]
    assert len(skipped) == 1
    assert skipped[0]["observation_quality"] == "incomplete"
    assert skipped[0]["computation_status"] == "unavailable"
    assert skipped[0]["missing_track_ids"] == ["1"]


def test_backward_source_steps_are_not_silently_sorted(source_run: Path, tmp_path: Path) -> None:
    rows = _signals(source_run)
    write_records_csv([*rows[2:], *rows[:2]], source_run / "agent_signals.csv")
    _refresh_artifacts(source_run)
    output = tmp_path / "backward-steps"
    with pytest.raises(ValueError, match="(?i)(nondecreasing|step)"):
        replay_run(ReplayOptions(source_run, output))
    assert not output.exists()


def test_inconsistent_source_time_is_not_silently_recomputed(
    source_run: Path, tmp_path: Path
) -> None:
    rows = _signals(source_run)
    rows[0]["time"] = "1234"
    write_records_csv(rows, source_run / "agent_signals.csv")
    _refresh_artifacts(source_run)
    output = tmp_path / "invalid-source-time"
    report = replay_run(ReplayOptions(source_run, output))
    assert report["status"] == "fail"
    assert len(_jsonl(output / "observations.jsonl")) == 3


def test_standalone_validator_rejects_truth_fields(source_run: Path, tmp_path: Path) -> None:
    output = tmp_path / "replayed"
    replay_run(ReplayOptions(source_run, output))
    rows = _jsonl(output / "observations.jsonl")
    rows[0]["observation"]["formation_truth"] = "not_an_input"
    candidate = tmp_path / "unexpected-field.jsonl"
    _write_jsonl(candidate, rows)
    report = validate_observation_file(candidate, metadata_path=output / "metadata.json")
    assert report["status"] == "fail"


def test_standalone_validator_rejects_decreasing_receipts(source_run: Path, tmp_path: Path) -> None:
    output = tmp_path / "replayed"
    replay_run(ReplayOptions(source_run, output))
    rows = _jsonl(output / "observations.jsonl")
    rows[0]["received_time"] = 1.0
    candidate = tmp_path / "backward-clock.jsonl"
    _write_jsonl(candidate, rows)
    report = validate_observation_file(candidate, metadata_path=output / "metadata.json")
    assert report["status"] == "fail"


def test_standalone_validator_rejects_early_delivery(source_run: Path, tmp_path: Path) -> None:
    output = tmp_path / "replayed"
    replay_run(ReplayOptions(source_run, output))
    rows = _jsonl(output / "observations.jsonl")
    rows[0]["received_time"] = rows[0]["observation"]["event_time"]
    candidate = tmp_path / "early.jsonl"
    _write_jsonl(candidate, rows)
    report = validate_observation_file(candidate, metadata_path=output / "metadata.json")
    assert report["status"] == "fail"


def test_standalone_validator_reports_wholly_missing_frame(
    source_run: Path, tmp_path: Path
) -> None:
    output = tmp_path / "replayed"
    replay_run(ReplayOptions(source_run, output))
    rows = [row for row in _jsonl(output / "observations.jsonl") if row["step"] == 1]
    candidate = tmp_path / "missing-frame.jsonl"
    _write_jsonl(candidate, rows)
    report = validate_observation_file(candidate, metadata_path=output / "metadata.json")
    assert report["status"] in {"fail", "degraded"}


def test_standalone_validator_rejects_empty_input(source_run: Path, tmp_path: Path) -> None:
    output = tmp_path / "replayed"
    replay_run(ReplayOptions(source_run, output))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    report = validate_observation_file(empty, metadata_path=output / "metadata.json")
    assert report["status"] == "fail"


@pytest.mark.parametrize("duplicate_scope", ["envelope", "observation"])
def test_standalone_validator_rejects_duplicate_json_keys(
    source_run: Path, tmp_path: Path, duplicate_scope: str
) -> None:
    output = tmp_path / "replayed"
    replay_run(ReplayOptions(source_run, output))
    rows = _jsonl(output / "observations.jsonl")
    first = json.dumps(rows[0])
    if duplicate_scope == "envelope":
        first = '{"received_time": 1234, ' + first[1:]
        duplicated_key = "received_time"
    else:
        first = first.replace(
            '"position_valid": true', '"position_valid": false, "position_valid": true'
        )
        duplicated_key = "position_valid"
    candidate = tmp_path / "duplicate-key.jsonl"
    candidate.write_text(
        first + "\n" + "".join(json.dumps(row) + "\n" for row in rows[1:]),
        encoding="utf-8",
    )
    report = validate_observation_file(candidate, metadata_path=output / "metadata.json")
    assert report["status"] == "fail"
    assert report["invalid_records"] == 1
    assert any(
        diagnostic.get("reason") == f"duplicate JSON key: {duplicated_key}"
        for diagnostic in report["diagnostics"]
    )


def test_standalone_validator_rejects_duplicate_metadata_keys(
    source_run: Path, tmp_path: Path
) -> None:
    output = tmp_path / "replayed"
    replay_run(ReplayOptions(source_run, output))
    metadata = _json(output / "metadata.json")
    candidate = tmp_path / "duplicate-metadata.json"
    candidate.write_text(
        '{"track_ids": ["untrusted"], ' + json.dumps(metadata)[1:], encoding="utf-8"
    )
    report = validate_observation_file(output / "observations.jsonl", metadata_path=candidate)
    assert report["status"] == "fail"
    assert report["records"] == 0
    assert report["diagnostics"][0]["reason"] == "duplicate JSON key: track_ids"


@pytest.mark.parametrize(
    "sample_id_case", ["reused_across_tracks", "reused_across_steps", "malformed"]
)
def test_standalone_validator_enforces_deterministic_sample_identity(
    source_run: Path, tmp_path: Path, sample_id_case: str
) -> None:
    output = tmp_path / "replayed"
    replay_run(ReplayOptions(source_run, output))
    rows = _jsonl(output / "observations.jsonl")
    if sample_id_case == "reused_across_tracks":
        rows[1]["observation"]["sample_id"] = rows[0]["observation"]["sample_id"]
    elif sample_id_case == "reused_across_steps":
        rows[2]["observation"]["sample_id"] = rows[0]["observation"]["sample_id"]
    else:
        rows[0]["observation"]["sample_id"] = "arbitrary-unbound-identity"
    candidate = tmp_path / "invalid-identity.jsonl"
    _write_jsonl(candidate, rows)
    report = validate_observation_file(candidate, metadata_path=output / "metadata.json")
    assert report["status"] == "fail"
    assert report["invalid_records"] == 1
    assert any(
        "sample_id disagrees" in diagnostic.get("reason", "")
        for diagnostic in report["diagnostics"]
    )


def test_standalone_validator_requires_cohort_array(source_run: Path, tmp_path: Path) -> None:
    output = tmp_path / "replayed"
    replay_run(ReplayOptions(source_run, output))
    metadata = _json(output / "metadata.json")
    # Iterating this string would accidentally recreate the two valid track IDs.
    metadata["track_ids"] = "12"
    candidate = tmp_path / "string-cohort.json"
    atomic_write_json(candidate, metadata)
    report = validate_observation_file(output / "observations.jsonl", metadata_path=candidate)
    assert report["status"] == "fail"
    assert report["records"] == 0
    assert report["diagnostics"][0]["reason"] == "metadata track_ids must be an array"


def test_csv_time_roundoff_is_validated_then_normalized(source_run: Path, tmp_path: Path) -> None:
    rows = _signals(source_run)
    for index, row in enumerate(rows):
        row["time"] = repr(float(row["time"]) + (index + 1) * 1e-12)
        row["next_time"] = repr(float(row["next_time"]) - (index + 1) * 1e-12)
    write_records_csv(rows, source_run / "agent_signals.csv")
    _refresh_artifacts(source_run)
    output = tmp_path / "roundoff"
    report = replay_run(ReplayOptions(source_run, output))
    assert report["status"] == "pass"
    assert report["counters"]["frames_ready"] == 2
    observations = _jsonl(output / "observations.jsonl")
    assert len(observations) == 4
    for envelope in observations:
        step = envelope["step"]
        assert envelope["observation"]["event_time"] == step * 0.25
        assert envelope["observation"]["available_time"] == (step + 1) * 0.25
        assert envelope["received_time"] == (step + 1) * 0.25
    assert (
        _json(output / "metadata.json")["source_time_roundoff_policy"]
        == "validate_then_normalize_to_declared_step_grid"
    )


def test_replay_and_validation_cli(source_run: Path, tmp_path: Path, capsys: Any) -> None:
    output = tmp_path / "cli-replay"
    assert main(["replay", str(source_run), "--output-directory", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert (
        main(
            [
                "validate-observations",
                str(output / "observations.jsonl"),
                "--metadata",
                str(output / "metadata.json"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "pass"


def test_replay_cli_reports_oversized_csv_field_without_traceback(
    source_run: Path, tmp_path: Path, capsys: Any
) -> None:
    rows = _signals(source_run)
    rows[0]["observed_state_position_x"] = "1" * 1025
    write_records_csv(rows, source_run / "agent_signals.csv")
    _refresh_artifacts(source_run)
    output = tmp_path / "rejected"
    original_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(1024)
        exit_code = main(["replay", str(source_run), "--output-directory", str(output)])
    finally:
        csv.field_size_limit(original_limit)
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["status"] == "fail"
    assert "field larger than field limit" in error["error"]
    assert "Traceback" not in captured.err
    assert not output.exists()
