"""Mesa agent implementation for individual 3D drones."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from mesa.experimental.continuous_space import ContinuousSpaceAgent  # type: ignore[import-untyped]
from numpy.typing import ArrayLike

from drone_swarm.mdp import DroneAction, DroneState, Transition, vector3
from drone_swarm.physics import (
    apply_boundary,
    as_vector3,
    clip_norm,
    integrate_kinematics,
    norm,
    steer_toward,
    unit_vector,
)
from drone_swarm.policies import ActionPolicy

if TYPE_CHECKING:  # pragma: no cover
    from drone_swarm.model import DroneSwarmModel


class DroneAgent(ContinuousSpaceAgent):  # type: ignore[misc]
    """A single drone with controller-facing and plant-facing transition logs."""

    def __init__(
        self,
        *,
        model: DroneSwarmModel,
        position: ArrayLike,
        velocity: ArrayLike,
        policy: ActionPolicy,
        agent_index: int,
        mode: str = "nominal",
        battery: float | None = None,
        coalition_id: str | None = "all",
        role: str | None = None,
        formation_id: str | None = None,
        target_position: ArrayLike | None = None,
        target_id: str | None = None,
        restrict_interactions_to_coalition: bool = False,
    ) -> None:
        super().__init__(space=model.space, model=model)
        self.model: DroneSwarmModel = model
        self.agent_index = int(agent_index)
        self.position = as_vector3(position, name="position")
        self.velocity = clip_norm(as_vector3(velocity, name="velocity"), model.max_speed)
        self.policy = policy
        self.mode = mode
        self.battery = battery
        self.coalition_id = coalition_id
        self.role = role
        self.formation_id = formation_id
        self.target_position = (
            as_vector3(target_position, name="target_position")
            if target_position is not None
            else None
        )
        self.target_id = target_id
        self.restrict_interactions_to_coalition = bool(restrict_interactions_to_coalition)

        self._true_s_t: DroneState | None = None
        self._s_t: DroneState | None = None
        self._a_t: DroneAction | None = None
        self._applied_a_t: DroneAction | None = None
        self._environment_acceleration: np.ndarray | None = None
        self._proposed_position: np.ndarray | None = None
        self._proposed_velocity: np.ndarray | None = None

    def _build_state(
        self,
        neighbors: Sequence[DroneAgent],
        distances: Sequence[float],
    ) -> DroneState:
        local_centroid = None
        local_average_velocity = None
        local_separation = None
        nearest_neighbor_distance = None

        if neighbors:
            positions = np.asarray([neighbor.position for neighbor in neighbors], dtype=float)
            velocities = np.asarray([neighbor.velocity for neighbor in neighbors], dtype=float)
            centroid = positions.mean(axis=0)
            average_velocity = velocities.mean(axis=0)
            local_centroid = vector3(centroid.tolist(), name="local_centroid")
            local_average_velocity = vector3(
                average_velocity.tolist(),
                name="local_average_velocity",
            )
            nearest_neighbor_distance = float(np.min(np.asarray(distances, dtype=float)))

            separation_direction = np.zeros(3, dtype=float)
            position = np.asarray(self.position, dtype=float)
            for neighbor, distance in zip(neighbors, distances, strict=False):
                distance_value = float(distance)
                if 0.0 < distance_value < self.model.separation_distance:
                    away = position - np.asarray(neighbor.position, dtype=float)
                    separation_direction += unit_vector(away) / max(distance_value, 1e-9)
            if np.linalg.norm(separation_direction) > 0:
                steering = steer_toward(
                    position,
                    self.velocity,
                    position + separation_direction,
                    self.model.max_speed,
                )
                local_separation = vector3(steering.tolist(), name="local_separation")

        target_vector = None
        if self.target_position is not None:
            target_vector = vector3(
                (self.target_position - np.asarray(self.position, dtype=float)).tolist(),
                name="target_vector",
            )

        return DroneState(
            position=vector3(np.asarray(self.position, dtype=float).tolist(), name="position"),
            velocity=vector3(np.asarray(self.velocity, dtype=float).tolist(), name="velocity"),
            speed=norm(self.velocity),
            neighbor_count=len(neighbors),
            nearest_neighbor_distance=nearest_neighbor_distance,
            local_centroid=local_centroid,
            local_average_velocity=local_average_velocity,
            neighbor_ids=tuple(int(neighbor.unique_id) for neighbor in neighbors),
            local_separation=local_separation,
            target_vector=target_vector,
            battery=self.battery,
            mode=self.mode,
        )

    def observe_true(self) -> DroneState:
        """Return the unperturbed physical state and nominal local neighborhood."""

        neighbors, distances = self.model.get_true_neighbors(self)
        return self._build_state(neighbors, distances)

    def observe(self, *, sample_key: str = "observation") -> DroneState:
        """Return the controller-facing observation after communication/sensor events."""

        neighbors, distances = self.model.get_perceived_neighbors(self)
        state = self._build_state(neighbors, distances)
        return self.model.event_manager.transform_observation(
            self.model,
            self,
            state,
            sample_key=sample_key,
        )

    def cache_state(self) -> None:
        """Cache true and observed ``S_t`` before any staged agent moves."""

        self._true_s_t = self.observe_true()
        self._s_t = self.observe(sample_key="pre")

    def select_action(self) -> None:
        """Select command and apply policy/actuator boundary events."""

        if self._s_t is None:
            raise RuntimeError("cache_state must run before select_action")
        nominal_command = self.policy.select_action(
            state=self._s_t,
            agent=self,
            model=self.model,
        )
        self._a_t = self.model.event_manager.transform_command(
            self.model,
            self,
            nominal_command,
        )
        self._applied_a_t = self.model.event_manager.transform_applied_action(
            self.model,
            self,
            self._a_t,
        )

    def propose_motion(self) -> None:
        """Compute proposed motion using applied action plus environmental force."""

        if self._applied_a_t is None:
            raise RuntimeError("select_action must run before propose_motion")
        self._environment_acceleration = self.model.event_manager.environment_acceleration(
            self.model,
            self,
        )
        total_acceleration = (
            np.asarray(self._applied_a_t.acceleration, dtype=float) + self._environment_acceleration
        )
        proposed_position, proposed_velocity = integrate_kinematics(
            self.position,
            self.velocity,
            total_acceleration,
            dt=self.model.dt,
            max_speed=self.model.max_speed,
        )
        proposed_position, proposed_velocity = apply_boundary(
            proposed_position,
            proposed_velocity,
            self.model.bounds,
            mode=self.model.boundary_mode,
        )

        self._proposed_position = proposed_position
        self._proposed_velocity = proposed_velocity

    def commit_motion(self) -> None:
        """Commit proposed motion to the Mesa continuous space."""

        if self._proposed_position is None or self._proposed_velocity is None:
            raise RuntimeError("propose_motion must run before commit_motion")

        self.velocity = self._proposed_velocity
        self.position = self._proposed_position

        if self.battery is not None:
            self.battery = max(0.0, self.battery - self.model.battery_drain_per_tick)
            if self.battery == 0.0:
                self.mode = "depleted"

    def log_transition(self) -> None:
        """Observe outcomes and emit one dual-view transition."""

        if (
            self._true_s_t is None
            or self._s_t is None
            or self._a_t is None
            or self._applied_a_t is None
            or self._environment_acceleration is None
        ):
            raise RuntimeError("cache_state, select_action and propose_motion must precede logging")

        true_s_prime = self.observe_true()
        s_prime = self.observe(sample_key="post")
        reward = self.model.reward_function(self._s_t, self._a_t, s_prime, self)
        done = self.model.done_function(self._s_t, self._a_t, s_prime, self)
        context = self.model.event_manager.transition_context(self.model, self)

        transition = Transition(
            episode_id=self.model.episode_id,
            step=self.model.tick,
            agent_id=int(self.unique_id),
            state=self._s_t,
            action=self._a_t,
            next_state=s_prime,
            reward=reward,
            done=done,
            true_state=self._true_s_t,
            applied_action=self._applied_a_t,
            environment_acceleration=vector3(self._environment_acceleration),
            true_next_state=true_s_prime,
            phase=context["phase"],
            active_event_ids=context["active_event_ids"],
            agent_affected=context["agent_affected"],
            coalition_truth=context["coalition_truth"],
            role_truth=context["role_truth"],
            formation_truth=context["formation_truth"],
            target_id=context["target_id"],
            info={"agent_index": self.agent_index},
        )
        self.model.record_transition(transition)

        self._true_s_t = None
        self._s_t = None
        self._a_t = None
        self._applied_a_t = None
        self._environment_acceleration = None
        self._proposed_position = None
        self._proposed_velocity = None

    def step_random(self) -> None:
        """Sequential random-activation step."""

        self.cache_state()
        self.select_action()
        self.propose_motion()
        self.commit_motion()
        self.log_transition()
