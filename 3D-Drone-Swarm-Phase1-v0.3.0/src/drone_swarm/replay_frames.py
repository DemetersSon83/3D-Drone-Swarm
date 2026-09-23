"""Bounded, causal, fixed-cohort observation admission (no feature calculations).

Frames are released in step order. A complete later frame immediately supersedes
unresolved earlier steps with explicit gap events. Deadlines admit arrivals at
the boundary itself unless their frame has already been finalized/superseded.
Identity/coordinate resets fail closed: create a new assembler for a new stream
segment. No interpolation, zero filling, or retrospective output revision occurs.
"""

from __future__ import annotations

import math
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from .observations import Observation, ObservationError


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass
class _Frame:
    observations: dict[str, Observation] = field(default_factory=dict)
    first_received: float = 0.0
    last_received: float = 0.0


class FrameAssembler:
    """Admit exact-step kinematic frames with bounded pending/duplicate storage.

    ``received_time`` and ``advance(now)`` use the declared observation clock,
    not a wall-clock duration. Both valid position and velocity are required.
    With ``expected_steps`` omitted, finish covers steps through the highest
    valid step number seen. Returned events are incremental and never retained.
    """

    def __init__(
        self,
        expected_track_ids: tuple[str, ...],
        dt: float,
        stream_id: str,
        availability_basis: str = "completed_transition_lower_bound",
        allowed_lateness: float | None = None,
        max_queued_frames: int = 4,
        max_tracks: int = 128,
        expected_steps: int | None = None,
    ) -> None:
        dt = _finite_number(dt, "dt")
        if dt <= 0:
            raise ValueError("dt must be finite and positive")
        lateness = (
            dt if allowed_lateness is None else _finite_number(allowed_lateness, "allowed_lateness")
        )
        if lateness < 0:
            raise ValueError("allowed_lateness must be finite and nonnegative")
        for value, name in ((max_tracks, "max_tracks"), (max_queued_frames, "max_queued_frames")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not expected_track_ids
            or any(not isinstance(track, str) or not track for track in expected_track_ids)
            or len(set(expected_track_ids)) != len(expected_track_ids)
            or len(expected_track_ids) > max_tracks
        ):
            raise ValueError("expected_track_ids must be unique nonempty strings within max_tracks")
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError("stream_id must be a nonempty string")
        if availability_basis not in {
            "completed_transition_lower_bound",
            "idealized_pre_state_replay",
        }:
            raise ValueError("fixed-step replay requires a declared replay availability basis")
        if expected_steps is not None and (
            isinstance(expected_steps, bool)
            or not isinstance(expected_steps, int)
            or expected_steps < 0
        ):
            raise ValueError("expected_steps must be a nonnegative integer")
        self.expected_track_ids = tuple(sorted(expected_track_ids))
        self.dt = dt
        self.stream_id = stream_id
        self.availability_basis = availability_basis
        self.allowed_lateness = lateness
        self.max_queued_frames = max_queued_frames
        self.expected_steps = expected_steps
        self._track_ids = frozenset(expected_track_ids)
        self._shift = int(availability_basis == "completed_transition_lower_bound")
        if expected_steps is not None and not self._valid_step_time(expected_steps):
            raise ValueError("expected_steps produces an unrepresentable time horizon")
        self._pending: dict[int, _Frame] = {}
        self._recent: OrderedDict[int, dict[str, Observation]] = OrderedDict()
        self._counts: Counter[str] = Counter()
        self._rejections: Counter[str] = Counter()
        self._skip_reasons: Counter[str] = Counter()
        self._next_step = 0
        self._highest_seen = -1
        self._now: float | None = None
        self._closed = False
        self._blocked = False
        self._signature: tuple[str, str, str, str] | None = None
        self._epochs: dict[str, int] = {}
        self._sources: dict[str, str] = {}
        self._generation = 0
        self._gap_pending = False
        self._gap_steps = 0

    def _nominal(self, step: int) -> float:
        return (step + self._shift) * self.dt

    def _deadline(self, step: int) -> float:
        return self._nominal(step) + self.allowed_lateness

    def _valid_step_time(self, step: int) -> bool:
        try:
            return math.isfinite(step * self.dt) and math.isfinite(self._deadline(step))
        except OverflowError:
            return False

    def _horizon(self) -> int:
        return self.expected_steps - 1 if self.expected_steps is not None else self._highest_seen

    def _clock(self, now: float) -> float:
        now = _finite_number(now, "receipt/advance time")
        if now < 0:
            raise ValueError("receipt/advance time must be finite and nonnegative")
        if self._now is not None and now < self._now:
            raise ValueError("receipt/advance time cannot decrease")
        self._now = now
        return now

    def _remember(self, step: int, frame: _Frame) -> None:
        self._recent[step] = frame.observations
        self._recent.move_to_end(step)
        while len(self._recent) > self.max_queued_frames:
            self._recent.popitem(last=False)

    def _reject(self, obs: Observation, step: int, reason: str, now: float) -> dict[str, Any]:
        self._counts["observations_rejected"] += 1
        self._rejections[reason] += 1
        if reason == "late_record":
            self._counts["late_records"] += 1
        if reason == "conflicting_duplicate":
            self._counts["conflicting_duplicates"] += 1
        return {
            "type": "observation_rejected",
            "reason": reason,
            "step": step,
            "sample_id": obs.sample_id,
            "track_id": obs.track_id,
            "emitted_time": now,
            "history_generation": self._generation,
        }

    def _skip(
        self,
        first: int,
        last: int,
        reason: str,
        now: float,
        frame: _Frame | None = None,
    ) -> dict[str, Any]:
        count = last - first + 1
        self._counts["frames_skipped"] += count
        self._skip_reasons[reason] += count
        self._gap_steps += count
        if not self._gap_pending:
            self._generation += 1
            self._gap_pending = True
        if frame is not None:
            self._remember(first, frame)
        self._next_step = last + 1
        observed = len(frame.observations) if frame is not None else 0
        return {
            "type": "frame_skipped",
            "reason": reason,
            "step": first if first == last else None,
            "first_step": first,
            "last_step": last,
            "frame_count": count,
            "observed_tracks": observed,
            "expected_tracks": len(self._track_ids),
            "missing_track_ids": [
                track
                for track in self.expected_track_ids
                if frame is None or track not in frame.observations
            ],
            "observation_quality": "incomplete",
            "computation_status": "unavailable",
            "timeliness": "deadline_missed" if now > self._deadline(first) else "within_deadline",
            "emitted_time": now,
            "history_generation": self._generation,
            "gap_since_previous_frame": True,
        }

    def _skip_through(self, last: int, reason: str, now: float) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while self._next_step <= last:
            step = self._next_step
            frame = self._pending.pop(step, None)
            if frame is not None:
                events.append(self._skip(step, step, reason, now, frame))
            else:
                next_present = min((s for s in self._pending if s > step), default=last + 1)
                events.append(self._skip(step, min(last, next_present - 1), reason, now))
        return events

    def _ready(self, step: int, frame: _Frame, now: float) -> dict[str, Any]:
        obs = [frame.observations[track] for track in self.expected_track_ids]
        latest = max(item.available_time for item in obs)
        # Informational flags are retained, but unknown covariance is not invalid data.
        degraded = any(item.is_imputed or item.is_extrapolated for item in obs)
        self._counts["frames_ready"] += 1
        self._remember(step, frame)
        self._next_step = step + 1
        result: dict[str, Any] = {
            "type": "frame_ready",
            "step": step,
            "event_time": step * self.dt,
            "nominal_available_time": self._nominal(step),
            "latest_input_availability": latest,
            "emitted_time": now,
            "first_received_time": frame.first_received,
            "last_received_time": frame.last_received,
            "queue_delay": now - frame.last_received,
            "input_delivery_delay": frame.last_received - latest,
            "sample_age": now - step * self.dt,
            "observations": [item.to_dict() for item in obs],
            "observation_quality": "degraded" if degraded else "valid",
            "uncertainty_status": "unknown"
            if any(
                item.position_covariance is None or item.velocity_covariance is None for item in obs
            )
            else "provided",
            "computation_status": "complete",
            "timeliness": "deadline_missed" if now > self._deadline(step) else "within_deadline",
            "history_generation": self._generation,
            "gap_since_previous_frame": self._gap_pending,
            "skipped_steps_since_previous_frame": self._gap_steps,
            "availability_basis": self.availability_basis,
        }
        self._gap_pending = False
        self._gap_steps = 0
        return result

    def _drain(self, now: float, *, finishing: bool = False) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        horizon = self._horizon()
        while self._next_step <= horizon:
            step = self._next_step
            frame = self._pending.get(step)
            if frame is not None and len(frame.observations) == len(self._track_ids):
                del self._pending[step]
                events.append(self._ready(step, frame, now))
                continue
            if not finishing and now <= self._deadline(step):
                break
            reason = "end_of_stream" if finishing else "incomplete_frame"
            if frame is not None:
                del self._pending[step]
                events.append(self._skip(step, step, reason, now, frame))
                continue
            # Skip absent ranges in one event; never allocate proportional to a time gap.
            next_present = min((s for s in self._pending if s > step), default=horizon + 1)
            last = min(horizon, next_present - 1)
            if not finishing and self._deadline(last) >= now:
                # Binary search is bounded by the bit length of the declared step range.
                lo, hi = step, last
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if self._deadline(mid) < now:
                        lo = mid
                    else:
                        hi = mid - 1
                last = lo
            events.append(self._skip(step, last, reason, now))
        return events

    def advance(self, now: float) -> list[dict[str, Any]]:
        """Expire incomplete steps strictly *after* their inclusive deadline."""
        if self._closed:
            raise ValueError("assembler is finished")
        now = self._clock(now)
        return self._drain(now)

    def _duplicate(self, obs: Observation) -> Observation | None:
        for frame in self._pending.values():
            for previous in frame.observations.values():
                if previous.sample_id == obs.sample_id:
                    return previous
        for observations in self._recent.values():
            for previous in observations.values():
                if previous.sample_id == obs.sample_id:
                    return previous
        return None

    def _reset(self, reason: str, step: int, now: float) -> list[dict[str, Any]]:
        self._blocked = True
        self._counts["history_resets"] += 1
        self._generation += 1
        self._gap_pending = True
        events: list[dict[str, Any]] = [
            {
                "type": "history_reset",
                "reason": reason,
                "step": step,
                "emitted_time": now,
                "history_generation": self._generation,
                "requires_new_assembler": True,
            }
        ]
        if self._pending:
            events.extend(self._skip_through(max(self._pending), reason, now))
        self._recent.clear()
        return events

    def push(self, obs: Observation, *, step: int, received_time: float) -> list[dict[str, Any]]:
        """Admit an observation, returning only newly issued events."""
        events = self.advance(received_time)
        received_time = float(received_time)
        self._counts["observations_received"] += 1
        try:
            obs.validate()
        except ObservationError:
            events.append(self._reject(obs, step, "invalid_observation", received_time))
            return events
        reason: str | None = None
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            reason = "invalid_step"
        elif self.expected_steps is not None and step >= self.expected_steps:
            reason = "step_out_of_bounds"
        elif not self._valid_step_time(step):
            reason = "invalid_step_time"
        elif obs.stream_id != self.stream_id:
            reason = "stream_mismatch"
        elif obs.track_id not in self._track_ids:
            reason = "unexpected_track"
        elif obs.availability_basis != self.availability_basis or obs.sample_phase != "pre_state":
            reason = "replay_mode_mismatch"
        elif not math.isclose(obs.event_time, step * self.dt, rel_tol=0.0, abs_tol=self.dt * 1e-8):
            reason = "event_time_mismatch"
        elif obs.available_time < self._nominal(step):
            reason = "before_nominal_availability"
        elif obs.available_time > received_time:
            reason = "future_observation"
        elif not obs.position_valid or not obs.velocity_valid:
            reason = "invalid_kinematics"
        elif self._blocked:
            reason = "stream_reset_required"
        if reason is not None:
            events.append(self._reject(obs, step, reason, received_time))
            return events
        self._highest_seen = max(self._highest_seen, step)
        events.extend(self._drain(received_time))
        previous = self._duplicate(obs)
        if previous is not None:
            if previous == obs:
                self._counts["duplicates_ignored"] += 1
                events.append(
                    {
                        "type": "duplicate_ignored",
                        "sample_id": obs.sample_id,
                        "step": step,
                        "emitted_time": received_time,
                    }
                )
            else:
                events.append(self._reject(obs, step, "conflicting_duplicate", received_time))
            return events
        if step < self._next_step:
            reason = (
                "conflicting_duplicate"
                if (step in self._recent and obs.track_id in self._recent[step])
                else "late_record"
            )
            events.append(self._reject(obs, step, reason, received_time))
            return events
        signature = (obs.clock_id, obs.time_unit, obs.frame_id, obs.length_unit)
        epoch = self._epochs.get(obs.track_id)
        if epoch is not None and obs.track_epoch < epoch:
            events.append(self._reject(obs, step, "stale_track_epoch", received_time))
            return events
        reset_reason = None
        if self._signature is not None and signature != self._signature:
            reset_reason = "frame_reset"
        elif (epoch is not None and obs.track_epoch != epoch) or (
            obs.track_id in self._sources and obs.source_id != self._sources[obs.track_id]
        ):
            reset_reason = "identity_reset"
        if reset_reason is not None:
            events.extend(self._reset(reset_reason, step, received_time))
            events.append(self._reject(obs, step, reset_reason, received_time))
            return events
        self._signature = signature
        self._epochs[obs.track_id] = obs.track_epoch
        self._sources[obs.track_id] = obs.source_id
        if step in self._pending and obs.track_id in self._pending[step].observations:
            events.append(self._reject(obs, step, "conflicting_duplicate", received_time))
            return events
        if step not in self._pending and len(self._pending) >= self.max_queued_frames:
            self._counts["capacity_exceeded"] += 1
            events.append(
                {
                    "type": "capacity_exceeded",
                    "reason": "capacity_exceeded",
                    "emitted_time": received_time,
                    "max_queued_frames": self.max_queued_frames,
                    "policy": "coalesce_to_newest_complete_else_skip_oldest",
                }
            )
            complete = [
                s for s, f in self._pending.items() if len(f.observations) == len(self._track_ids)
            ]
            if complete:
                events.extend(
                    self._skip_through(max(complete) - 1, "capacity_exceeded", received_time)
                )
                events.extend(self._drain(received_time))
            if len(self._pending) >= self.max_queued_frames:
                events.extend(
                    self._skip_through(min(self._pending), "capacity_exceeded", received_time)
                )
            if step < self._next_step:
                events.append(self._reject(obs, step, "late_record", received_time))
                return events
        frame = self._pending.setdefault(step, _Frame(first_received=received_time))
        frame.observations[obs.track_id] = obs
        frame.last_received = received_time
        self._counts["observations_admitted"] += 1
        self._counts["pending_high_water"] = max(
            self._counts["pending_high_water"], len(self._pending)
        )
        self._counts["pending_observations_high_water"] = max(
            self._counts["pending_observations_high_water"],
            sum(len(f.observations) for f in self._pending.values()),
        )
        if len(frame.observations) == len(self._track_ids) and step > self._next_step:
            events.extend(
                self._skip_through(step - 1, "superseded_by_complete_frame", received_time)
            )
        events.extend(self._drain(received_time))
        return events

    def finish(self) -> list[dict[str, Any]]:
        """Finalize remaining frames, including wholly absent declared steps."""
        if self._closed:
            return []
        events = self._drain(0.0 if self._now is None else self._now, finishing=True)
        self._closed = True
        return events

    def summary(self) -> dict[str, Any]:
        """Return bounded diagnostics without retaining emitted event history."""
        names = (
            "observations_received",
            "observations_admitted",
            "observations_rejected",
            "duplicates_ignored",
            "conflicting_duplicates",
            "late_records",
            "frames_ready",
            "frames_skipped",
            "capacity_exceeded",
            "history_resets",
            "pending_high_water",
            "pending_observations_high_water",
        )
        return {
            **{name: self._counts[name] for name in names},
            "rejection_reasons": dict(self._rejections),
            "skip_reasons": dict(self._skip_reasons),
            "history_generation": self._generation,
            "next_step": self._next_step,
            "pending_frames": len(self._pending),
            "pending_observations": sum(len(f.observations) for f in self._pending.values()),
            "cached_finalized_frames": len(self._recent),
            "cached_finalized_observations": sum(len(f) for f in self._recent.values()),
            "requires_new_assembler": self._blocked,
            "finished": self._closed,
        }
