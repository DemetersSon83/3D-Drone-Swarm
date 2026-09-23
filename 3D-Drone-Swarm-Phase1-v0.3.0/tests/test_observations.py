"""Observation-contract boundaries, timing, and detector-payload isolation."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from drone_swarm.observations import Observation, ObservationError, observation_from_signal


def signal_row() -> dict[str, Any]:
    return {
        "contract_version": "1.0.0",
        "run_id": 'run:1["quoted"]',
        "step": 2,
        "agent_id": 3,
        "time": 0.1,
        "next_time": 0.15,
        "observed_state_position_x": 1.0,
        "observed_state_position_y": 2.0,
        "observed_state_position_z": 3.0,
        "observed_state_velocity_x": -1.0,
        "observed_state_velocity_y": 0.0,
        "observed_state_velocity_z": 1.0,
    }


def adapt(row: dict[str, Any] | None = None, **kwargs: Any) -> Observation:
    row = signal_row() if row is None else row
    return observation_from_signal(
        row,
        run_id='run:1["quoted"]',
        dt=0.05,
        source_id="simulator",
        frame_id="simulation_cartesian",
        **kwargs,
    )


def test_replay_maps_only_pre_state_and_declares_unknown_uncertainty() -> None:
    observation = adapt()
    assert observation.position == (1.0, 2.0, 3.0)
    assert observation.velocity == (-1.0, 0.0, 1.0)
    assert observation.event_time == 0.1
    assert observation.available_time == 3 * 0.05
    assert observation.sample_phase == "pre_state"
    assert observation.track_id == "3"
    assert observation.track_epoch == 0
    assert observation.position_covariance is None
    assert observation.velocity_covariance is None
    assert "completed_transition_lower_bound" in observation.quality_flags
    assert not observation.is_imputed
    assert not observation.is_extrapolated
    assert json.loads(observation.sample_id) == ['run:1["quoted"]', 2, "3", "pre_state"]


def test_truth_actions_next_state_and_future_labels_cannot_change_payload() -> None:
    original = signal_row()
    changed = dict(original)
    for name in (
        "true_state_position_x",
        "true_next_state_velocity_x",
        "observed_next_state_position_x",
        "observed_next_state_velocity_y",
        "applied_action_acceleration_x",
        "environment_acceleration_z",
        "phase",
        "active_event_ids",
        "agent_affected",
        "coalition_truth",
        "role_truth",
        "formation_truth",
        "target_id",
        "reward",
        "scenario_id",
        "commanded_action_type",
        "observed_state_neighbor_ids",
    ):
        changed[name] = {"arbitrary": float("nan")}
    assert adapt(original).to_dict() == adapt(changed).to_dict()


def test_idealized_mode_is_explicit_and_does_not_admit_next_state() -> None:
    completed = adapt()
    idealized = adapt(availability_basis="idealized_pre_state_replay")
    assert idealized.available_time == idealized.event_time
    assert idealized.availability_basis != completed.availability_basis
    assert idealized.quality_flags != completed.quality_flags
    assert idealized.position == completed.position
    with pytest.raises(ObservationError, match="assumed availability"):
        adapt(availability_basis="measured")


def test_json_round_trip_is_strict_and_nested_arrays_are_immutable() -> None:
    observation = adapt()
    encoded = json.loads(json.dumps(observation.to_dict(), allow_nan=False))
    parsed = Observation.from_dict(encoded)
    assert parsed == observation
    encoded["position"][0] = 99
    encoded["quality_flags"].append("mutated")
    assert parsed.position == (1.0, 2.0, 3.0)
    assert "mutated" not in parsed.quality_flags
    with pytest.raises(FrozenInstanceError):
        parsed.track_epoch = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    ("name", "bad"),
    [
        ("event_time", "0.1"),
        ("available_time", True),
        ("available_time", float("inf")),
        ("position_valid", 1),
        ("velocity_valid", "true"),
        ("is_imputed", 0),
        ("is_extrapolated", "false"),
        ("track_id", 3),
        ("track_epoch", 0.0),
        ("track_epoch", True),
        ("track_epoch", -1),
        ("source_id", ""),
        ("frame_id", " "),
        ("observation_contract_version", "0.2.0"),
        ("availability_basis", "probably_measured"),
        ("sample_phase", "post_state"),
        ("position", [0.0, float("nan"), 0.0]),
        ("velocity", [0.0, float("inf"), 0.0]),
        ("position", [0, True, 1]),
        ("position", [0, 1]),
        ("position", "123"),
        ("quality_flags", "assumed"),
        ("quality_flags", ["x", "x"]),
        ("quality_flags", [1]),
    ],
)
def test_canonical_types_and_values_are_not_silently_coerced(name: str, bad: Any) -> None:
    value = adapt().to_dict()
    value[name] = bad
    with pytest.raises(ObservationError) as caught:
        Observation.from_dict(value)
    assert caught.value.code == "invalid_observation"


def test_exact_field_set_rejects_missing_and_extra_metadata() -> None:
    value = adapt().to_dict()
    del value["is_imputed"]
    with pytest.raises(ObservationError, match="missing observation fields"):
        Observation.from_dict(value)
    value = adapt().to_dict()
    value["formation_truth"] = "unused"
    with pytest.raises(ObservationError, match="unknown observation fields"):
        Observation.from_dict(value)


def test_availability_boundary_and_invalid_measurement_semantics() -> None:
    observation = adapt()
    replace(observation, available_time=observation.event_time).validate()
    with pytest.raises(ObservationError, match="at least"):
        replace(observation, available_time=observation.event_time - 1e-12)
    with pytest.raises(ObservationError, match="null vector"):
        replace(observation, position=None)
    missing = replace(observation, position=None, position_valid=False)
    missing.validate()
    flagged = replace(observation, position_valid=False)
    assert flagged.position is not None
    assert not flagged.position_valid


def test_direct_construction_validates_and_validate_detects_forced_mutation() -> None:
    value = adapt().to_dict()
    value["event_time"] = float("nan")
    with pytest.raises(ObservationError, match="finite"):
        Observation(**value)
    observation = adapt()
    object.__setattr__(observation, "position", (float("nan"), 0.0, 0.0))
    with pytest.raises(ObservationError, match="finite"):
        observation.validate()


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_covariance_accepts_scaled_semidefinite_and_roundoff_tolerance(scale: float) -> None:
    covariance = [[scale, scale, 0.0], [scale, scale, 0.0], [0.0, 0.0, 0.0]]
    value = adapt().to_dict()
    value["position_covariance"] = covariance
    observation = Observation.from_dict(value)
    assert observation.position_covariance is not None
    covariance[0][0] = 0
    assert observation.position_covariance[0][0] == scale
    value["position_covariance"] = [[scale, 1e-12 * scale, 0], [0, scale, 0], [0, 0, 0]]
    Observation.from_dict(value).validate()


@pytest.mark.parametrize(
    "covariance",
    [
        [[1, 0], [0, 1]],
        [[1, 0, 0], None, [0, 0, 1]],
        [[1, 0, 0], [1, 1, 0], [0, 0, 1]],
        [[1, 2, 0], [2, 1, 0], [0, 0, 1]],
        [[1e-200, 0, 0], [0, -1e-200, 0], [0, 0, 0]],
        [[1, 0, 0], [0, float("nan"), 0], [0, 0, 1]],
        [[True, 0, 0], [0, 1, 0], [0, 0, 1]],
    ],
)
def test_covariance_rejects_wrong_shape_asymmetry_and_negative_energy(covariance: Any) -> None:
    value = adapt().to_dict()
    value["velocity_covariance"] = covariance
    with pytest.raises(ObservationError):
        Observation.from_dict(value)


def test_zero_covariance_is_distinct_from_unknown_covariance() -> None:
    unknown = adapt()
    zero = replace(unknown, position_covariance=((0.0, 0.0, 0.0),) * 3)
    assert unknown.to_dict()["position_covariance"] is None
    assert zero.to_dict()["position_covariance"] == [[0, 0, 0]] * 3
    with pytest.raises(ObservationError, match="requires a supplied"):
        replace(zero, position=None, position_valid=False)


def test_csv_source_coercion_is_local_and_sample_identity_is_unambiguous() -> None:
    row = signal_row()
    for key in ("step", "agent_id", "time", "next_time"):
        row[key] = str(row[key])
    for key in row:
        if key.startswith("observed_state_"):
            row[key] = str(row[key])
    assert adapt(row) == adapt()
    row["agent_id"] = "03"
    with pytest.raises(ObservationError, match="canonical"):
        adapt(row)


@pytest.mark.parametrize(
    ("name", "bad"),
    [
        ("contract_version", "0.9.0"),
        ("run_id", "other"),
        ("step", True),
        ("step", 2.0),
        ("step", -1),
        ("step", "2.0"),
        ("agent_id", False),
        ("agent_id", "3.0"),
        ("time", 0.05),
        ("time", "nan"),
        ("time", True),
        ("next_time", 0.1),
        ("next_time", 0.16),
        ("observed_state_position_x", float("nan")),
        ("observed_state_velocity_y", float("inf")),
    ],
)
def test_source_identity_and_timing_fail_closed(name: str, bad: Any) -> None:
    row = signal_row()
    row[name] = bad
    with pytest.raises(ObservationError):
        adapt(row)


def test_source_partial_missing_rejected_whole_vector_missing_is_explicit() -> None:
    row = signal_row()
    row["observed_state_velocity_x"] = None
    with pytest.raises(ObservationError, match="partially missing"):
        adapt(row)
    for axis in "xyz":
        row[f"observed_state_velocity_{axis}"] = None
    missing = adapt(row)
    assert missing.velocity is None
    assert not missing.velocity_valid
    assert "velocity_missing" in missing.quality_flags
    del row["observed_state_position_x"]
    with pytest.raises(ObservationError, match="missing source fields"):
        adapt(row)


def test_large_clock_does_not_relax_time_check_to_fraction_of_frame() -> None:
    row = signal_row()
    row["step"] = 100_000_000
    row["time"] = row["step"] * 0.05
    row["next_time"] = (row["step"] + 1) * 0.05
    adapt(row).validate()
    row["time"] += 0.001
    with pytest.raises(ObservationError, match="step \\* dt"):
        adapt(row)
