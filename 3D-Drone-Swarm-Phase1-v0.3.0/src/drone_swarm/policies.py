"""Action policies for 3D boids-like drone behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from drone_swarm.mdp import ActionType, DroneAction, Vector3, vector3
from drone_swarm.physics import (
    boundary_avoidance_acceleration,
    clip_norm,
    steer_toward,
)

if TYPE_CHECKING:  # pragma: no cover
    from drone_swarm.drone import DroneAgent
    from drone_swarm.mdp import DroneState
    from drone_swarm.model import DroneSwarmModel


def _to_vector3_tuple(value: np.ndarray) -> Vector3:
    return vector3(value.tolist())


class ActionPolicy(Protocol):
    """Protocol for policies that map a logged ``DroneState`` to an action."""

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

    Policy randomness uses the model's policy-specific RNG stream, which is
    independent from initialization and perturbation streams.
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
        raw = model.policy_rng.normal(loc=0.0, scale=self.scale, size=3)
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

    The policy intentionally consumes only the supplied, logged ``DroneState``.
    It no longer queries hidden neighbor objects.  ``local_separation`` is the
    sufficient steering vector computed by the observation layer.
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
        del agent

        position = np.asarray(state.position, dtype=float)
        velocity = np.asarray(state.velocity, dtype=float)
        has_neighbors = state.neighbor_count > 0

        components: dict[str, Vector3] = {}
        acceleration = np.zeros(3, dtype=float)

        if has_neighbors and state.local_centroid is not None:
            cohesion = steer_toward(
                position,
                velocity,
                np.asarray(state.local_centroid, dtype=float),
                model.max_speed,
            )
        else:
            cohesion = np.zeros(3, dtype=float)

        if has_neighbors and state.local_average_velocity is not None:
            alignment = np.asarray(state.local_average_velocity, dtype=float) - velocity
        else:
            alignment = np.zeros(3, dtype=float)

        separation = (
            np.asarray(state.local_separation, dtype=float)
            if state.local_separation is not None
            else np.zeros(3, dtype=float)
        )

        components["cohesion"] = _to_vector3_tuple(cohesion)
        components["alignment"] = _to_vector3_tuple(alignment)
        components["separation"] = _to_vector3_tuple(separation)
        acceleration += self.cohesion_weight * cohesion
        acceleration += self.alignment_weight * alignment
        acceleration += self.separation_weight * separation

        if state.target_vector is not None:
            target = position + np.asarray(state.target_vector, dtype=float)
            goal = steer_toward(position, velocity, target, model.max_speed)
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

        action_type: ActionType = "boids_steer"
        if np.linalg.norm(boundary) > 0:
            action_type = "avoid_boundary"
        elif (
            has_neighbors
            and state.nearest_neighbor_distance is not None
            and 0.0 < state.nearest_neighbor_distance < model.separation_distance
        ):
            action_type = "avoid_neighbor"
        elif state.target_vector is not None:
            action_type = "return_to_base"

        return DroneAction(
            acceleration=_to_vector3_tuple(acceleration),
            action_type=action_type,
            clipped=clipped,
            raw_acceleration=_to_vector3_tuple(raw_acceleration),
            components=components,
        )
