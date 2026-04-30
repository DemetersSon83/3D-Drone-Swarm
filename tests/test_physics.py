from __future__ import annotations

import numpy as np
import pytest

from drone_swarm.physics import apply_boundary, as_bounds3, as_vector3, clip_norm, integrate_kinematics


def test_as_vector3_validates_shape() -> None:
    assert np.allclose(as_vector3([1, 2, 3]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        as_vector3([1, 2])


def test_as_bounds3_validates_shape_and_order() -> None:
    bounds = as_bounds3([[0, 1], [0, 2], [-1, 1]])
    assert bounds.shape == (3, 2)
    with pytest.raises(ValueError):
        as_bounds3([[0, 1], [0, 2]])
    with pytest.raises(ValueError):
        as_bounds3([[1, 0], [0, 2], [-1, 1]])


def test_clip_norm_preserves_small_vectors_and_clips_large_vectors() -> None:
    assert np.allclose(clip_norm([3, 4, 0], 10), np.array([3, 4, 0], dtype=float))
    clipped = clip_norm([3, 4, 0], 2)
    assert pytest.approx(np.linalg.norm(clipped)) == 2


def test_integrate_kinematics_clips_speed() -> None:
    next_position, next_velocity = integrate_kinematics(
        [0, 0, 0],
        [10, 0, 0],
        [0, 0, 0],
        dt=1.0,
        max_speed=2.0,
    )
    assert np.allclose(next_velocity, [2, 0, 0])
    assert np.allclose(next_position, [2, 0, 0])


def test_apply_boundary_clip() -> None:
    position, velocity = apply_boundary(
        [-1, 5, 11],
        [-2, 0, 3],
        [[0, 10], [0, 10], [0, 10]],
        mode="clip",
    )
    assert np.allclose(position, [0, 5, 10])
    assert np.allclose(velocity, [0, 0, 0])


def test_apply_boundary_bounce() -> None:
    position, velocity = apply_boundary(
        [-1, 5, 11],
        [-2, 0, 3],
        [[0, 10], [0, 10], [0, 10]],
        mode="bounce",
    )
    assert np.allclose(position, [1, 5, 9])
    assert np.allclose(velocity, [2, 0, -3])


def test_apply_boundary_wrap() -> None:
    position, velocity = apply_boundary(
        [-1, 5, 11],
        [-2, 0, 3],
        [[0, 10], [0, 10], [0, 10]],
        mode="wrap",
    )
    assert np.allclose(position, [9, 5, 1])
    assert np.allclose(velocity, [-2, 0, 3])
