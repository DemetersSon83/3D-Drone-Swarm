"""3D vector and kinematics helpers for the swarm simulator."""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
BoundaryMode: TypeAlias = Literal["clip", "bounce", "wrap"]


def as_vector3(value: ArrayLike, *, name: str = "vector") -> FloatArray:
    """Convert *value* to a finite NumPy vector of shape ``(3,)``."""

    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(float, copy=True)


def as_bounds3(bounds: ArrayLike) -> FloatArray:
    """Convert bounds to a finite ``(3, 2)`` NumPy array."""

    bounds_array = np.asarray(bounds, dtype=float)
    if bounds_array.shape != (3, 2):
        raise ValueError(f"bounds must have shape (3, 2), got {bounds_array.shape}")
    if not np.isfinite(bounds_array).all():
        raise ValueError("bounds must contain only finite values")
    if not np.all(bounds_array[:, 1] > bounds_array[:, 0]):
        raise ValueError("each bounds row must satisfy max > min")
    return bounds_array.astype(float, copy=True)


def norm(vector: ArrayLike) -> float:
    """Return the Euclidean norm of *vector*."""

    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def unit_vector(vector: ArrayLike, *, eps: float = 1e-12) -> FloatArray:
    """Return a unit vector, or zeros when the vector is near zero."""

    vector_array = np.asarray(vector, dtype=float)
    magnitude = np.linalg.norm(vector_array)
    if magnitude <= eps:
        return np.zeros_like(vector_array, dtype=float)
    return (vector_array / magnitude).astype(float, copy=False)


def clip_norm(vector: ArrayLike, max_norm: float) -> FloatArray:
    """Clip *vector* to ``max_norm`` while preserving direction."""

    if max_norm < 0:
        raise ValueError("max_norm must be non-negative")
    vector_array = np.asarray(vector, dtype=float)
    magnitude = np.linalg.norm(vector_array)
    if magnitude == 0 or magnitude <= max_norm:
        return vector_array.astype(float, copy=True)
    return (vector_array / magnitude * max_norm).astype(float, copy=False)


def steer_toward(
    current_position: ArrayLike,
    current_velocity: ArrayLike,
    target_position: ArrayLike,
    max_speed: float,
) -> FloatArray:
    """Return a steering vector that points from current state toward a target."""

    desired_direction = unit_vector(np.asarray(target_position) - np.asarray(current_position))
    desired_velocity = desired_direction * max_speed
    return desired_velocity - np.asarray(current_velocity, dtype=float)


def integrate_kinematics(
    position: ArrayLike,
    velocity: ArrayLike,
    acceleration: ArrayLike,
    *,
    dt: float,
    max_speed: float,
) -> tuple[FloatArray, FloatArray]:
    """Integrate one Euler step and return ``(next_position, next_velocity)``."""

    if dt <= 0:
        raise ValueError("dt must be positive")
    if max_speed < 0:
        raise ValueError("max_speed must be non-negative")

    position_array = as_vector3(position, name="position")
    velocity_array = as_vector3(velocity, name="velocity")
    acceleration_array = as_vector3(acceleration, name="acceleration")

    next_velocity = clip_norm(velocity_array + acceleration_array * dt, max_speed)
    next_position = position_array + next_velocity * dt
    return next_position, next_velocity


def apply_boundary(
    position: ArrayLike,
    velocity: ArrayLike,
    bounds: ArrayLike,
    *,
    mode: BoundaryMode = "bounce",
) -> tuple[FloatArray, FloatArray]:
    """Apply boundary handling to a proposed position and velocity.

    ``clip`` clamps positions to the box and removes velocity into the wall.
    ``bounce`` reflects positions and velocities off the box walls.
    ``wrap`` wraps positions toroidally and leaves velocity unchanged.
    """

    position_array = as_vector3(position, name="position")
    velocity_array = as_vector3(velocity, name="velocity")
    bounds_array = as_bounds3(bounds)

    low = bounds_array[:, 0]
    high = bounds_array[:, 1]
    width = high - low

    if mode == "wrap":
        wrapped_position = low + np.mod(position_array - low, width)
        return wrapped_position, velocity_array

    if mode == "clip":
        clipped_position = np.clip(position_array, low, high)
        clipped_velocity = velocity_array.copy()
        for axis in range(3):
            if position_array[axis] < low[axis] and clipped_velocity[axis] < 0:
                clipped_velocity[axis] = 0.0
            elif position_array[axis] > high[axis] and clipped_velocity[axis] > 0:
                clipped_velocity[axis] = 0.0
        return clipped_position, clipped_velocity

    if mode != "bounce":
        raise ValueError(f"unsupported boundary mode: {mode}")

    bounced_position = position_array.copy()
    bounced_velocity = velocity_array.copy()

    for axis in range(3):
        guard = 0
        while bounced_position[axis] < low[axis] or bounced_position[axis] > high[axis]:
            if bounced_position[axis] < low[axis]:
                bounced_position[axis] = low[axis] + (low[axis] - bounced_position[axis])
                bounced_velocity[axis] = -bounced_velocity[axis]
            elif bounced_position[axis] > high[axis]:
                bounced_position[axis] = high[axis] - (bounced_position[axis] - high[axis])
                bounced_velocity[axis] = -bounced_velocity[axis]

            guard += 1
            if guard > 10_000:  # pragma: no cover - pathological overshoot protection
                bounced_position[axis] = low[axis] + np.mod(bounced_position[axis] - low[axis], width[axis])
                break

    return bounced_position, bounced_velocity


def boundary_avoidance_acceleration(
    position: ArrayLike,
    bounds: ArrayLike,
    *,
    margin: float,
    strength: float,
) -> FloatArray:
    """Return an inward acceleration when near the simulation boundaries."""

    if margin <= 0 or strength == 0:
        return np.zeros(3, dtype=float)

    position_array = as_vector3(position, name="position")
    bounds_array = as_bounds3(bounds)
    low = bounds_array[:, 0]
    high = bounds_array[:, 1]

    acceleration = np.zeros(3, dtype=float)
    for axis in range(3):
        distance_to_low = position_array[axis] - low[axis]
        distance_to_high = high[axis] - position_array[axis]
        if distance_to_low < margin:
            acceleration[axis] += strength * (1.0 - max(distance_to_low, 0.0) / margin)
        if distance_to_high < margin:
            acceleration[axis] -= strength * (1.0 - max(distance_to_high, 0.0) / margin)

    return acceleration
