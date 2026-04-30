"""Action policies for 3D boids-like drone behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from drone_swarm.mdp import DroneAction, Vector3, vector3
from drone_swarm.physics import (
    boundary_avoidance_acceleration,
    clip_norm,
    steer_toward,
    unit_vector,
)

if TYPE_CHECKING:  # pragma: no cover
    from drone_swarm.drone import DroneAgent
    from drone_swarm.mdp import DroneState
    from drone_swarm.model import DroneSwarmModel


def _to_vector3_tuple(value: np.ndarray) -> Vector3:
    return vector3(value.tolist())


class ActionPolicy(Protocol):
    """Protocol for policies that map ``state`` to an action."""

    def select_action(
        self,
        *,
        state: DroneState,
        agent: DroneAgent,
        model: DroneSwarmModel,
    ) -> DroneAction:
        """Select an action for *agent* under *state*."""


@dataclass(frozen=True, slots=True)
class HoldPolicy:
    """A policy that commands zero acceleration."""

    def select_action(
        self,
        *,
        state: DroneState,
        agent: DroneAgent,
        model: DroneSwarmModel,
    ) -> DroneAction:
        del state, agent, model
        return DroneAction(acceleration=(0.0, 0.0, 0.0), action_type="hold")


@dataclass(frozen=True, slots=True)
class RandomAccelerationPolicy:
    """Random exploratory acceleration policy.

    This is useful for testing transition capture and baseline comparisons.
    """

    scale: float = 1.0

    def select_action(
        self,
        *,
        state: DroneState,
        agent: DroneAgent,
        model: DroneSwarmModel,
    ) -> DroneAction:
        del state, agent
        raw = model.np_random.normal(loc=0.0, scale=self.scale, size=3)
        clipped = np.linalg.norm(raw) > model.max_acceleration
        acceleration = clip_norm(raw, model.max_acceleration)
        return DroneAction(
            acceleration=_to_vector3_tuple(acceleration),
            action_type="random_acceleration",
            clipped=bool(clipped),
            raw_acceleration=_to_vector3_tuple(raw),
        )


@dataclass(frozen=True, slots=True)
class BoidsPolicy:
    """Continuous 3D boids steering policy.

    The selected action is a bounded acceleration vector composed of interpretable
    steering components: cohesion, alignment, separation, goal seeking, and
    boundary avoidance.
    """

    cohesion_weight: float = 0.025
    alignment_weight: float = 0.05
    separation_weight: float = 0.25
    goal_weight: float = 0.02
    boundary_weight: float = 1.0

    def select_action(
        self,
        *,
        state: DroneState,
        agent: DroneAgent,
        model: DroneSwarmModel,
    ) -> DroneAction:
        del state

        neighbors, distances = agent.get_neighbors_in_radius(model.perception_radius)
        position = np.asarray(agent.position, dtype=float)
        velocity = np.asarray(agent.velocity, dtype=float)

        components: dict[str, Vector3] = {}
        acceleration = np.zeros(3, dtype=float)

        if neighbors:
            neighbor_positions = np.asarray([neighbor.position for neighbor in neighbors], dtype=float)
            neighbor_velocities = np.asarray([neighbor.velocity for neighbor in neighbors], dtype=float)

            centroid = neighbor_positions.mean(axis=0)
            avg_velocity = neighbor_velocities.mean(axis=0)

            cohesion = steer_toward(position, velocity, centroid, model.max_speed)
            alignment = avg_velocity - velocity

            separation = np.zeros(3, dtype=float)
            for neighbor, distance in zip(neighbors, distances, strict=False):
                if 0.0 < distance < model.separation_distance:
                    away = position - np.asarray(neighbor.position, dtype=float)
                    separation += unit_vector(away) / max(distance, 1e-9)
            if np.linalg.norm(separation) > 0:
                separation = steer_toward(
                    position,
                    velocity,
                    position + separation,
                    model.max_speed,
                )

            components["cohesion"] = _to_vector3_tuple(cohesion)
            components["alignment"] = _to_vector3_tuple(alignment)
            components["separation"] = _to_vector3_tuple(separation)

            acceleration += self.cohesion_weight * cohesion
            acceleration += self.alignment_weight * alignment
            acceleration += self.separation_weight * separation
        else:
            components["cohesion"] = (0.0, 0.0, 0.0)
            components["alignment"] = (0.0, 0.0, 0.0)
            components["separation"] = (0.0, 0.0, 0.0)

        if model.target_position is not None:
            goal = steer_toward(position, velocity, model.target_position, model.max_speed)
            components["goal"] = _to_vector3_tuple(goal)
            acceleration += self.goal_weight * goal
        else:
            components["goal"] = (0.0, 0.0, 0.0)

        boundary = boundary_avoidance_acceleration(
            position,
            model.bounds,
            margin=model.boundary_margin,
            strength=model.max_acceleration,
        )
        components["boundary"] = _to_vector3_tuple(boundary)
        acceleration += self.boundary_weight * boundary

        raw_acceleration = acceleration.copy()
        clipped = bool(np.linalg.norm(raw_acceleration) > model.max_acceleration)
        acceleration = clip_norm(raw_acceleration, model.max_acceleration)

        action_type = "boids_steer"
        if np.linalg.norm(boundary) > 0:
            action_type = "avoid_boundary"
        elif neighbors and any(0.0 < distance < model.separation_distance for distance in distances):
            action_type = "avoid_neighbor"
        elif model.target_position is not None:
            action_type = "return_to_base"

        return DroneAction(
            acceleration=_to_vector3_tuple(acceleration),
            action_type=action_type,
            clipped=clipped,
            raw_acceleration=_to_vector3_tuple(raw_acceleration),
            components=components,
        )
