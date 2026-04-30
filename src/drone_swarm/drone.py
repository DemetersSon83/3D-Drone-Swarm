"""Mesa agent implementation for individual 3D drones."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mesa.experimental.continuous_space import ContinuousSpaceAgent

from drone_swarm.mdp import DroneAction, DroneState, Transition, vector3
from drone_swarm.physics import as_vector3, apply_boundary, clip_norm, integrate_kinematics, norm
from drone_swarm.policies import ActionPolicy

if TYPE_CHECKING:  # pragma: no cover
    from drone_swarm.model import DroneSwarmModel


class DroneAgent(ContinuousSpaceAgent):
    """A single drone with 3D position and velocity.

    Agents follow a staged lifecycle under the default model activation mode:
    cache state, select action, propose motion, commit motion, and log the MDP
    transition. This keeps every transition aligned to one global model tick.
    """

    def __init__(
        self,
        *,
        model: DroneSwarmModel,
        position: object,
        velocity: object,
        policy: ActionPolicy,
        mode: str = "nominal",
        battery: float | None = None,
    ) -> None:
        super().__init__(space=model.space, model=model)
        self.position = as_vector3(position, name="position")
        self.velocity = clip_norm(as_vector3(velocity, name="velocity"), model.max_speed)
        self.policy = policy
        self.mode = mode
        self.battery = battery

        self._s_t: DroneState | None = None
        self._a_t: DroneAction | None = None
        self._proposed_position: np.ndarray | None = None
        self._proposed_velocity: np.ndarray | None = None

    def observe(self) -> DroneState:
        """Build the current Markov state for this drone."""

        neighbors, distances = self.get_neighbors_in_radius(self.model.perception_radius)

        local_centroid = None
        local_average_velocity = None
        nearest_neighbor_distance = None
        if neighbors:
            positions = np.asarray([neighbor.position for neighbor in neighbors], dtype=float)
            velocities = np.asarray([neighbor.velocity for neighbor in neighbors], dtype=float)
            centroid = positions.mean(axis=0)
            avg_velocity = velocities.mean(axis=0)
            local_centroid = vector3(centroid.tolist(), name="local_centroid")
            local_average_velocity = vector3(avg_velocity.tolist(), name="local_average_velocity")
            nearest_neighbor_distance = float(np.min(distances))

        target_vector = None
        if self.model.target_position is not None:
            target_vector = vector3(
                (self.model.target_position - np.asarray(self.position, dtype=float)).tolist(),
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
            target_vector=target_vector,
            battery=self.battery,
            mode=self.mode,
        )

    def cache_state(self) -> None:
        """Cache ``S_t`` before any agent moves this tick."""

        self._s_t = self.observe()

    def select_action(self) -> None:
        """Select and cache ``A_t`` using the cached state."""

        if self._s_t is None:
            raise RuntimeError("cache_state must run before select_action")
        self._a_t = self.policy.select_action(state=self._s_t, agent=self, model=self.model)

    def propose_motion(self) -> None:
        """Compute proposed position and velocity without committing them."""

        if self._a_t is None:
            raise RuntimeError("select_action must run before propose_motion")

        proposed_position, proposed_velocity = integrate_kinematics(
            self.position,
            self.velocity,
            self._a_t.acceleration,
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
        """Observe ``S_{t+1}`` and append a full transition to the model log."""

        if self._s_t is None or self._a_t is None:
            raise RuntimeError("cache_state and select_action must run before log_transition")

        s_prime = self.observe()
        reward = self.model.reward_function(self._s_t, self._a_t, s_prime, self)
        done = self.model.done_function(self._s_t, self._a_t, s_prime, self)

        self.model.transition_log.append(
            Transition(
                episode_id=self.model.episode_id,
                step=self.model.tick,
                agent_id=int(self.unique_id),
                state=self._s_t,
                action=self._a_t,
                next_state=s_prime,
                reward=reward,
                done=done,
            )
        )

        self._s_t = None
        self._a_t = None
        self._proposed_position = None
        self._proposed_velocity = None

    def step_random(self) -> None:
        """Sequential random-activation step.

        This mode is available for experiments, but staged activation is the
        default because it yields cleaner same-tick MDP transitions.
        """

        self.cache_state()
        self.select_action()
        self.propose_motion()
        self.commit_motion()
        self.log_transition()
