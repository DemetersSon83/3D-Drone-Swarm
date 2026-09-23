"""Causal timing, missingness, reset, and resource contract tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from drone_swarm.observations import Observation
from drone_swarm.replay_frames import FrameAssembler


def observation(step: int = 0, track: str = "a", **updates: Any) -> Observation:
    obs = Observation(
        observation_contract_version="0.1.0",
        stream_id="run",
        source_id="simulator",
        sample_id=f"run:{step}:{track}:pre_state",
        track_id=track,
        track_epoch=0,
        event_time=step * 0.05,
        available_time=(step + 1) * 0.05,
        clock_id="simulation",
        time_unit="simulation_time",
        availability_basis="completed_transition_lower_bound",
        sample_phase="pre_state",
        frame_id="simulation_cartesian",
        length_unit="simulation_length",
        position=(1.0, 2.0, 3.0),
        position_valid=True,
        velocity=(0.0, 0.0, 0.0),
        velocity_valid=True,
        velocity_source="simulator_controller_observation",
        position_covariance=None,
        velocity_covariance=None,
        quality_flags=("covariance_unknown",),
        is_imputed=False,
        is_extrapolated=False,
    )
    return replace(obs, **updates) if updates else obs


def assembler(**updates: Any) -> FrameAssembler:
    args: dict[str, Any] = {
        "expected_track_ids": ("b", "a"),
        "dt": 0.05,
        "stream_id": "run",
        "expected_steps": 3,
    }
    args.update(updates)
    return FrameAssembler(**args)


def of_type(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [event for event in events if event["type"] == kind]


def test_complete_emits_at_receipt_deterministically_with_no_hold() -> None:
    frames = assembler()
    assert frames.push(observation(track="b"), step=0, received_time=0.05) == []
    events = frames.push(observation(), step=0, received_time=0.05)
    ready = of_type(events, "frame_ready")[0]
    assert [o["track_id"] for o in ready["observations"]] == ["a", "b"]
    assert ready["emitted_time"] == ready["latest_input_availability"] == 0.05
    assert ready["event_time"] == 0.0
    assert ready["observation_quality"] == "valid"
    assert ready["uncertainty_status"] == "unknown"
    assert ready["queue_delay"] == 0.0
    assert ready["gap_since_previous_frame"] is False


def test_lateness_starts_at_availability_and_boundary_is_inclusive() -> None:
    frames = assembler()
    frames.push(observation(), step=0, received_time=0.05)
    assert frames.advance(0.099) == []
    assert frames.advance(0.1) == []
    events = frames.push(observation(track="b"), step=0, received_time=0.1)
    assert len(of_type(events, "frame_ready")) == 1
    assert frames.summary()["frames_skipped"] == 0


def test_cross_step_interleaving_preserves_prior_inclusive_deadline() -> None:
    frames = assembler()
    frames.push(observation(), step=0, received_time=0.05)
    assert frames.push(observation(1), step=1, received_time=0.1) == []
    events = frames.push(observation(0, "b"), step=0, received_time=0.1)
    assert [e["step"] for e in of_type(events, "frame_ready")] == [0]
    events = frames.push(observation(1, "b"), step=1, received_time=0.1)
    assert [e["step"] for e in of_type(events, "frame_ready")] == [1]
    assert frames.summary()["frames_skipped"] == 0


def test_complete_later_frame_emits_immediately_with_explicit_superseding_gap() -> None:
    frames = assembler()
    frames.push(observation(), step=0, received_time=0.05)
    frames.push(observation(1), step=1, received_time=0.1)
    events = frames.push(observation(1, "b"), step=1, received_time=0.1)
    skipped = of_type(events, "frame_skipped")[0]
    ready = of_type(events, "frame_ready")[0]
    assert skipped["step"] == 0
    assert skipped["reason"] == "superseded_by_complete_frame"
    assert ready["step"] == 1
    assert ready["emitted_time"] == 0.1
    assert ready["queue_delay"] == 0.0
    assert ready["gap_since_previous_frame"] is True
    assert frames.push(observation(0, "b"), step=0, received_time=0.1)[0]["reason"] == "late_record"
    assert frames.summary()["frames_ready"] == 1


def test_partial_and_whole_missing_frames_have_visible_gap() -> None:
    frames = assembler(expected_steps=4)
    frames.push(observation(), step=0, received_time=0.05)
    events = frames.advance(0.151)
    skipped = of_type(events, "frame_skipped")
    assert [(e["first_step"], e["last_step"]) for e in skipped] == [(0, 0), (1, 1)]
    assert skipped[0]["missing_track_ids"] == ["b"]
    assert skipped[1]["missing_track_ids"] == ["a", "b"]
    frames.push(observation(2), step=2, received_time=0.151)
    ready = frames.push(observation(2, "b"), step=2, received_time=0.151)[0]
    assert ready["gap_since_previous_frame"] is True
    assert ready["skipped_steps_since_previous_frame"] == 2
    assert ready["history_generation"] == 1
    remaining = frames.finish()
    assert of_type(remaining, "frame_skipped")[0]["first_step"] == 3
    assert frames.summary()["frames_ready"] + frames.summary()["frames_skipped"] == 4


def test_finish_reports_absent_declared_steps_in_constant_space() -> None:
    frames = assembler(expected_steps=10**8)
    events = frames.finish()
    assert len(events) == 1
    assert events[0]["first_step"] == 0
    assert events[0]["last_step"] == 10**8 - 1
    assert events[0]["frame_count"] == 10**8
    assert frames.finish() == []


def test_advance_large_gap_is_compact() -> None:
    frames = assembler(expected_steps=10**8)
    events = frames.advance(10**8)
    assert len(events) == 1
    assert events[0]["frame_count"] == 10**8


def test_duplicate_and_conflict_before_and_after_finalization() -> None:
    frames = assembler()
    first = observation()
    frames.push(first, step=0, received_time=0.05)
    assert frames.push(first, step=0, received_time=0.05)[0]["type"] == "duplicate_ignored"
    changed = replace(first, position=(9.0, 2.0, 3.0))
    assert frames.push(changed, step=0, received_time=0.05)[0]["reason"] == "conflicting_duplicate"
    frames.push(observation(track="b"), step=0, received_time=0.05)
    assert frames.push(first, step=0, received_time=0.05)[0]["type"] == "duplicate_ignored"
    assert frames.push(changed, step=0, received_time=0.05)[0]["reason"] == "conflicting_duplicate"
    assert frames.summary()["duplicates_ignored"] == 2
    assert frames.summary()["conflicting_duplicates"] == 2


def test_second_sample_identity_for_same_track_step_is_conflict() -> None:
    frames = assembler()
    frames.push(observation(), step=0, received_time=0.05)
    events = frames.push(observation(sample_id="another"), step=0, received_time=0.05)
    assert events[0]["reason"] == "conflicting_duplicate"
    frames.push(observation(track="b"), step=0, received_time=0.05)
    events = frames.push(observation(sample_id="another"), step=0, received_time=0.05)
    assert events[0]["reason"] == "conflicting_duplicate"


def test_late_record_cannot_rewrite_skipped_or_ready_frame() -> None:
    frames = assembler()
    frames.push(observation(), step=0, received_time=0.05)
    frames.advance(0.101)
    events = frames.push(observation(track="b"), step=0, received_time=0.101)
    assert events[0]["reason"] == "late_record"
    assert frames.summary()["frames_ready"] == 0
    assert frames.summary()["late_records"] == 1


def test_future_sample_rejected_and_no_invalid_zero_frame() -> None:
    frames = assembler()
    events = frames.push(observation(available_time=0.06), step=0, received_time=0.05)
    assert events[0]["reason"] == "future_observation"
    invalid = observation(velocity=None, velocity_valid=False)
    assert frames.push(invalid, step=0, received_time=0.05)[0]["reason"] == "invalid_kinematics"
    assert frames.summary()["observations_admitted"] == 0
    assert frames.summary()["frames_ready"] == 0
    assert frames.finish()[0]["computation_status"] == "unavailable"


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"available_time": 0.0}, "before_nominal_availability"),
        ({"event_time": 0.001}, "event_time_mismatch"),
        ({"track_id": "unexpected"}, "unexpected_track"),
        ({"stream_id": "another"}, "stream_mismatch"),
        ({"availability_basis": "measured"}, "replay_mode_mismatch"),
    ],
)
def test_admission_bounds(changes: dict[str, Any], reason: str) -> None:
    frames = assembler()
    assert frames.push(observation(**changes), step=0, received_time=0.05)[0]["reason"] == reason


def test_steps_and_clock_bounds() -> None:
    frames = assembler()
    assert frames.push(observation(), step=-1, received_time=0.05)[0]["reason"] == "invalid_step"
    assert (
        frames.push(observation(), step=3, received_time=0.05)[0]["reason"] == "step_out_of_bounds"
    )
    with pytest.raises(ValueError, match="cannot decrease"):
        frames.push(observation(), step=0, received_time=0.04)
    with pytest.raises(ValueError, match="finite"):
        frames.advance(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        frames.advance(True)
    unbounded = assembler(expected_steps=None)
    event = unbounded.push(observation(), step=10**500, received_time=0.05)[0]
    assert event["reason"] == "invalid_step_time"


def test_idealized_mode_has_separate_availability_boundary() -> None:
    frames = assembler(availability_basis="idealized_pre_state_replay")
    for track in ("a", "b"):
        events = frames.push(
            observation(
                track=track, available_time=0.0, availability_basis="idealized_pre_state_replay"
            ),
            step=0,
            received_time=0.0,
        )
    ready = events[0]
    assert ready["nominal_available_time"] == ready["emitted_time"] == 0.0
    assert ready["availability_basis"] == "idealized_pre_state_replay"


def test_capacity_and_newest_complete_frame_have_explicit_skips() -> None:
    frames = assembler(expected_steps=8, max_queued_frames=3, allowed_lateness=1.0)
    frames.push(observation(0), step=0, received_time=0.05)
    frames.push(observation(1), step=1, received_time=0.1)
    frames.push(observation(2), step=2, received_time=0.151)
    events = frames.push(observation(3), step=3, received_time=0.2)
    assert len(of_type(events, "capacity_exceeded")) == 1
    assert of_type(events, "frame_skipped")[0]["step"] == 0
    assert of_type(events, "frame_skipped")[0]["reason"] == "capacity_exceeded"
    events = frames.push(observation(3, "b"), step=3, received_time=0.2)
    assert [event["step"] for event in of_type(events, "frame_ready")] == [3]
    assert sum(event["frame_count"] for event in of_type(events, "frame_skipped")) == 2
    assert all(
        event["reason"] == "superseded_by_complete_frame"
        for event in of_type(events, "frame_skipped")
    )
    assert frames.summary()["pending_high_water"] <= 3
    assert frames.summary()["capacity_exceeded"] == 1


def test_capacity_without_complete_frame_evicts_oldest_explicitly() -> None:
    frames = assembler(expected_steps=4, max_queued_frames=1, allowed_lateness=1.0)
    frames.push(observation(0), step=0, received_time=0.05)
    events = frames.push(observation(1), step=1, received_time=0.1)
    assert of_type(events, "frame_skipped")[0]["reason"] == "capacity_exceeded"
    assert frames.summary()["pending_frames"] == 1
    assert frames.summary()["pending_high_water"] == 1


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"track_epoch": 1}, "identity_reset"),
        ({"source_id": "new_producer"}, "identity_reset"),
        ({"frame_id": "new_frame"}, "frame_reset"),
        ({"clock_id": "new_clock"}, "frame_reset"),
        ({"length_unit": "m"}, "frame_reset"),
    ],
)
def test_reset_invalidates_partials_and_fails_closed(changes: dict[str, Any], reason: str) -> None:
    frames = assembler(allowed_lateness=1.0)
    frames.push(observation(0), step=0, received_time=0.05)
    events = frames.push(observation(1, **changes), step=1, received_time=0.1)
    assert of_type(events, "history_reset")[0]["reason"] == reason
    assert frames.summary()["pending_frames"] == 0
    assert frames.summary()["requires_new_assembler"] is True
    assert (
        frames.push(observation(1, "b"), step=1, received_time=0.1)[0]["reason"]
        == "stream_reset_required"
    )
    assert frames.summary()["frames_ready"] == 0
    assert frames.summary()["history_resets"] == 1


def test_cache_and_pending_memory_remain_bounded_over_many_frames() -> None:
    frames = assembler(expected_steps=1000, max_queued_frames=4)
    for step in range(1000):
        now = (step + 1) * 0.05
        frames.push(observation(step, "b"), step=step, received_time=now)
        frames.push(observation(step), step=step, received_time=now)
    summary = frames.summary()
    assert summary["frames_ready"] == 1000
    assert summary["pending_frames"] == 0
    assert summary["cached_finalized_frames"] == 4
    assert summary["cached_finalized_observations"] == 8
    assert summary["pending_observations_high_water"] <= 8
    assert frames.push(observation(), step=0, received_time=50.0)[0]["reason"] == "late_record"


def test_quality_and_gap_axes_are_independent() -> None:
    frames = assembler(expected_track_ids=("a",))
    events = frames.push(observation(is_imputed=True), step=0, received_time=0.05)
    assert events[0]["observation_quality"] == "degraded"
    assert events[0]["computation_status"] == "complete"
    assert events[0]["timeliness"] == "within_deadline"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0},
        {"dt": float("inf")},
        {"allowed_lateness": -1},
        {"dt": True},
        {"dt": "0.05"},
        {"dt": 10**500},
        {"allowed_lateness": True},
        {"allowed_lateness": "0.05"},
        {"max_queued_frames": 0},
        {"max_queued_frames": True},
        {"expected_steps": -1},
        {"expected_steps": True},
        {"expected_steps": 10**500},
        {"expected_track_ids": ("a", "a")},
        {"expected_track_ids": ()},
        {"max_tracks": 1},
        {"availability_basis": "measured"},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        assembler(**kwargs)
