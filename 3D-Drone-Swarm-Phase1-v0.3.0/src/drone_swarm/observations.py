"""Strict, observation-only records for causal simulator replay.

The adapter deliberately reads only observed pre-state kinematics and the source
identity/time fields. It never consumes truth, actions, labels, or next-state
observations. Logged rows do not measure delivery latency: their default
availability is the declared completed-transition lower bound.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from numbers import Integral, Real
from typing import Any

import numpy as np

OBSERVATION_CONTRACT_VERSION = "0.1.0"
SOURCE_CONTRACT_VERSION = "1.0.0"
AVAILABILITY_BASES = frozenset(
    {"measured", "completed_transition_lower_bound", "idealized_pre_state_replay"}
)
REPLAY_AVAILABILITY_BASES = AVAILABILITY_BASES - {"measured"}
SAMPLE_PHASES = frozenset({"pre_state"})
Vector3 = tuple[float, float, float]
Covariance3 = tuple[Vector3, Vector3, Vector3]


class ObservationError(ValueError):
    """A rejected observation or source row with a stable machine-readable code."""

    code = "invalid_observation"


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObservationError(f"{name} must be a nonempty string")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ObservationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ObservationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ObservationError(f"{name} must be a finite number")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ObservationError(f"{name} must be a nonnegative integer")
    return int(value)


def _vector(value: Any, name: str) -> Vector3 | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ObservationError(f"{name} must be null or a three-element numeric array")
    return (
        _number(value[0], f"{name}[0]"),
        _number(value[1], f"{name}[1]"),
        _number(value[2], f"{name}[2]"),
    )


def _covariance(value: Any, name: str) -> Covariance3 | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ObservationError(f"{name} must be null or a 3-by-3 numeric array")
    rows = tuple(_vector(row, f"{name}[{i}]") for i, row in enumerate(value))
    if any(row is None for row in rows):
        raise ObservationError(f"{name} must contain three numeric rows")
    # Scale first so large finite values cannot overflow the symmetry/eigenvalue
    # checks. The 1e-10 tolerance is relative to the maximum absolute entry;
    # small covariances do not receive an arbitrary unit-sized absolute floor.
    matrix = np.asarray(rows, dtype=float)
    scale = float(np.max(np.abs(matrix)))
    if scale:
        scaled = matrix / scale
        if not np.allclose(scaled, scaled.T, rtol=0.0, atol=1e-10):
            raise ObservationError(f"{name} must be symmetric")
        try:
            eigenvalues = np.linalg.eigvalsh(0.5 * scaled + 0.5 * scaled.T)
        except np.linalg.LinAlgError as exc:
            raise ObservationError(f"{name} could not be validated") from exc
        if float(eigenvalues[0]) < -1e-10:
            raise ObservationError(f"{name} must be positive semidefinite")
    # The checks above exclude None without discarding precise tuple typing.
    first, second, third = rows
    assert first is not None and second is not None and third is not None
    return first, second, third


@dataclass(frozen=True)
class Observation:
    """Immutable canonical observation; construction and parsing are validated.

    A null vector requires a false validity flag. A finite supplied vector may
    still be marked invalid, preserving an explicitly rejected measurement for
    quality reporting. No spatial calculation should admit invalid positions.
    Unknown covariance is represented by None, never a zero matrix.
    """

    observation_contract_version: str
    stream_id: str
    source_id: str
    sample_id: str
    track_id: str
    track_epoch: int
    event_time: float
    available_time: float
    clock_id: str
    time_unit: str
    availability_basis: str
    sample_phase: str
    frame_id: str
    length_unit: str
    position: Vector3 | None
    position_valid: bool
    velocity: Vector3 | None
    velocity_valid: bool
    velocity_source: str
    position_covariance: Covariance3 | None
    velocity_covariance: Covariance3 | None
    quality_flags: tuple[str, ...]
    is_imputed: bool
    is_extrapolated: bool

    def __post_init__(self) -> None:
        canonical = _validated_fields(self._field_dict())
        for name, value in canonical.items():
            object.__setattr__(self, name, value)

    def _field_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def validate(self) -> None:
        """Validate direct-constructed records, including every nested value."""
        _validated_fields(self._field_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Observation:
        """Parse exact contract fields without coercing JSON strings or booleans."""
        if not isinstance(value, Mapping):
            raise ObservationError("observation must be an object")
        names = {field.name for field in fields(cls)}
        if any(not isinstance(key, str) for key in value):
            raise ObservationError("observation field names must be strings")
        missing = names - value.keys()
        unknown = value.keys() - names
        if missing:
            raise ObservationError(f"missing observation fields: {', '.join(sorted(missing))}")
        if unknown:
            raise ObservationError(f"unknown observation fields: {', '.join(sorted(unknown))}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible representation."""
        self.validate()
        result = self._field_dict()
        for name in ("position", "velocity"):
            vector = result[name]
            result[name] = None if vector is None else list(vector)
        for name in ("position_covariance", "velocity_covariance"):
            covariance = result[name]
            result[name] = None if covariance is None else [list(row) for row in covariance]
        result["quality_flags"] = list(self.quality_flags)
        return result


def _validated_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for name in (
        "observation_contract_version",
        "stream_id",
        "source_id",
        "sample_id",
        "track_id",
        "clock_id",
        "time_unit",
        "availability_basis",
        "sample_phase",
        "frame_id",
        "length_unit",
        "velocity_source",
    ):
        result[name] = _string(result[name], name)
    if result["observation_contract_version"] != OBSERVATION_CONTRACT_VERSION:
        raise ObservationError("unsupported observation_contract_version")
    if result["availability_basis"] not in AVAILABILITY_BASES:
        raise ObservationError("unsupported availability_basis")
    if result["sample_phase"] not in SAMPLE_PHASES:
        raise ObservationError("unsupported sample_phase")
    result["track_epoch"] = _integer(result["track_epoch"], "track_epoch")
    for name in ("event_time", "available_time"):
        result[name] = _number(result[name], name)
    if result["available_time"] < result["event_time"]:
        raise ObservationError("available_time must be at least event_time")
    for name in ("position_valid", "velocity_valid", "is_imputed", "is_extrapolated"):
        if not isinstance(result[name], bool):
            raise ObservationError(f"{name} must be a boolean")
    for name in ("position", "velocity"):
        result[name] = _vector(result[name], name)
        if result[name] is None and result[f"{name}_valid"]:
            raise ObservationError(f"{name}_valid cannot be true for a null vector")
        covariance_name = f"{name}_covariance"
        result[covariance_name] = _covariance(result[covariance_name], covariance_name)
        if result[name] is None and result[covariance_name] is not None:
            raise ObservationError(f"{covariance_name} requires a supplied {name}")
    quality = result["quality_flags"]
    if not isinstance(quality, (list, tuple)):
        raise ObservationError("quality_flags must be an array of strings")
    result["quality_flags"] = tuple(_string(flag, "quality_flags entry") for flag in quality)
    if len(set(result["quality_flags"])) != len(result["quality_flags"]):
        raise ObservationError("quality_flags must be unique")
    return result


def _source_number(value: Any, name: str) -> float:
    """Accept numeric CSV text only at the source adapter boundary."""
    if isinstance(value, str):
        try:
            value = float(value)
        except (ValueError, OverflowError) as exc:
            raise ObservationError(f"source {name} must be a finite number") from exc
    return _number(value, f"source {name}")


def _source_integer(value: Any, name: str) -> int:
    if isinstance(value, str):
        if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
            raise ObservationError(f"source {name} must be a canonical nonnegative integer")
        try:
            value = int(value)
        except ValueError as exc:
            raise ObservationError(f"source {name} is out of range") from exc
    return _integer(value, f"source {name}")


def _source_vector(row: Mapping[str, Any], prefix: str) -> Vector3 | None:
    values = [row[f"{prefix}_{axis}"] for axis in "xyz"]
    missing = [value is None or (isinstance(value, str) and value == "") for value in values]
    if all(missing):
        return None
    if any(missing):
        raise ObservationError(f"source {prefix} has partially missing components")
    return (
        _source_number(values[0], f"{prefix}_x"),
        _source_number(values[1], f"{prefix}_y"),
        _source_number(values[2], f"{prefix}_z"),
    )


def observation_from_signal(
    row: Mapping[str, Any],
    *,
    run_id: str,
    dt: float,
    source_id: str,
    frame_id: str,
    availability_basis: str = "completed_transition_lower_bound",
    clock_id: str = "simulation",
    length_unit: str = "simulation_length",
    time_unit: str = "simulation_time",
) -> Observation:
    """Extract a pre-state observation from a contract-1.0.0 agent-signals row.

    Source CSV numeric text is accepted here; canonical records remain strictly
    typed. A surrounding reader must verify the complete artifact schema/hash.
    This function intentionally accesses only the required allowlisted fields.
    """
    if not isinstance(row, Mapping):
        raise ObservationError("source row must be an object")
    run_id = _string(run_id, "run_id")
    dt = _number(dt, "dt")
    if dt <= 0:
        raise ObservationError("dt must be positive")
    availability_basis = _string(availability_basis, "availability_basis")
    if availability_basis not in REPLAY_AVAILABILITY_BASES:
        raise ObservationError("simulator replay requires an explicit assumed availability basis")
    required = {"contract_version", "run_id", "step", "agent_id", "time", "next_time"}
    required.update(
        f"observed_state_{kind}_{axis}" for kind in ("position", "velocity") for axis in "xyz"
    )
    missing = required - row.keys()
    if missing:
        raise ObservationError(f"missing source fields: {', '.join(sorted(missing))}")
    if row["contract_version"] != SOURCE_CONTRACT_VERSION:
        raise ObservationError("unsupported source contract_version")
    if row["run_id"] != run_id:
        raise ObservationError("source run_id does not match the declared run")
    step = _source_integer(row["step"], "step")
    agent_id = _source_integer(row["agent_id"], "agent_id")
    event_time = _source_number(row["time"], "time")
    next_time = _source_number(row["next_time"], "next_time")
    try:
        expected_time = step * dt
        expected_next_time = (step + 1) * dt
    except OverflowError as exc:
        raise ObservationError("source step and dt produce unrepresentable times") from exc
    if not math.isfinite(expected_time) or not math.isfinite(expected_next_time):
        raise ObservationError("source step and dt produce nonfinite times")
    if expected_next_time <= expected_time:
        raise ObservationError("source time precision cannot represent the declared dt")
    # A few floating-point roundoff units are tolerated, with an additional
    # dt-relative floor for serialized decimal arithmetic. Never tolerate a
    # sizable fraction of a frame solely because the absolute clock is large.
    tolerance = min(dt * 1e-7, max(dt * 1e-10, 8 * math.ulp(expected_next_time)))
    if abs(event_time - expected_time) > tolerance:
        raise ObservationError("source time must equal step * dt")
    if abs(next_time - expected_next_time) > tolerance:
        raise ObservationError("source next_time must equal (step + 1) * dt")
    if next_time <= event_time:
        raise ObservationError("source next_time must be later than time")
    # After checking source times, use the exact declared grid so harmless CSV
    # roundoff cannot place an observation just before its nominal availability.
    event_time, next_time = expected_time, expected_next_time
    position = _source_vector(row, "observed_state_position")
    velocity = _source_vector(row, "observed_state_velocity")
    flags = [availability_basis, "covariance_unknown"]
    if position is None:
        flags.append("position_missing")
    if velocity is None:
        flags.append("velocity_missing")
    return Observation(
        observation_contract_version=OBSERVATION_CONTRACT_VERSION,
        stream_id=run_id,
        source_id=source_id,
        sample_id=json.dumps([run_id, step, str(agent_id), "pre_state"], separators=(",", ":")),
        track_id=str(agent_id),
        track_epoch=0,
        event_time=event_time,
        available_time=next_time
        if availability_basis == "completed_transition_lower_bound"
        else event_time,
        clock_id=clock_id,
        time_unit=time_unit,
        availability_basis=availability_basis,
        sample_phase="pre_state",
        frame_id=frame_id,
        length_unit=length_unit,
        position=position,
        position_valid=position is not None,
        velocity=velocity,
        velocity_valid=velocity is not None,
        velocity_source="simulator_controller_observation",
        position_covariance=None,
        velocity_covariance=None,
        quality_flags=tuple(flags),
        is_imputed=False,
        is_extrapolated=False,
    )
