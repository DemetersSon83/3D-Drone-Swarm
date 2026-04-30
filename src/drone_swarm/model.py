"""Mesa model for a 3D boids-style drone swarm."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias
from uuid import uuid4

import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.experimental.continuous_space import ContinuousSpace
from numpy.typing import ArrayLike, NDArray

from drone_swarm.drone import DroneAgent
from drone_swarm.io import write_transitions_csv, write_transitions_jsonl, write_transitions_parquet
from drone_swarm.mdp import DroneAction, DroneState, Transition
from drone_swarm.metrics import collision_count, mean_speed, min_pairwise_distance, swarm_centroid
from drone_swarm.physics import BoundaryMode, as_bounds3, as_vector3, clip_norm
from drone_swarm.policies import ActionPolicy, BoidsPolicy

ActivationMode: TypeAlias = Literal["staged", "random"]
RewardFunction: TypeAlias = Callable[[DroneState, DroneAction, DroneState, DroneAgent], float | None]
DoneFunction: TypeAlias = Callable[[DroneState, DroneAction, DroneState, DroneAgent], bool]
PositionSampler: TypeAlias = Callable[[np.random.Generator, NDArray[np.float64]], NDArray[np.float64]]
VelocitySampler: TypeAlias = Callable[[np.random.Generator, float], NDArray[np.float64]]


def _default_position_sampler(
    rng: np.random.Generator,
    bounds: NDArray[np.float64],
) -> NDArray[np.float64]:
    low = bounds[:, 0]
    high = bounds[:, 1]
    return rng.uniform(low=low, high=high, size=3).astype(float)


def _default_velocity_sampler(
    rng: np.random.Generator,
    max_speed: float,
) -> NDArray[np.float64]:
    direction = rng.normal(size=3)
    if np.linalg.norm(direction) == 0:
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
    direction = direction / np.linalg.norm(direction)
    speed = rng.uniform(0.0, max_speed)
    return (direction * speed).astype(float)


def _no_reward(
    state: DroneState,
    action: DroneAction,
    next_state: DroneState,
    agent: DroneAgent,
) -> None:
    del state, action, next_state, agent
    return None


def _never_done(
    state: DroneState,
    action: DroneAction,
    next_state: DroneState,
    agent: DroneAgent,
) -> bool:
    del state, action, next_state, agent
    return False


class DroneSwarmModel(Model):
    """A 3D Mesa model for boids-like drone swarm behavior.

    Parameters are intentionally explicit so runs can be reproduced and logged.
    The default activation mode is ``staged``: all drones sense the same world at
    time ``t``, all choose actions, all propose motion, all commit motion, and
    then all log ``(S_t, A_t, S_{t+1})``.
    """

    def __init__(
        self,
        *,
        n_drones: int = 50,
        bounds: ArrayLike = ((0.0, 100.0), (0.0, 100.0), (0.0, 50.0)),
        seed: int | None = None,
        dt: float = 1.0,
        perception_radius: float = 10.0,
        separation_distance: float = 2.0,
        collision_radius: float = 1.0,
        max_speed: float = 3.0,
        max_acceleration: float = 0.5,
        torus: bool = False,
        boundary_mode: BoundaryMode = "bounce",
        boundary_margin: float = 5.0,
        target_position: ArrayLike | None = None,
        policy: ActionPolicy | None = None,
        activation: ActivationMode = "staged",
        battery_initial: float | None = None,
        battery_drain_per_tick: float = 0.0,
        reward_function: RewardFunction | None = None,
        done_function: DoneFunction | None = None,
        initial_position_sampler: PositionSampler | None = None,
        initial_velocity_sampler: VelocitySampler | None = None,
        collect_data: bool = True,
        episode_id: str | None = None,
    ) -> None:
        super().__init__(seed=seed)

        if n_drones < 0:
            raise ValueError("n_drones must be non-negative")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if perception_radius <= 0:
            raise ValueError("perception_radius must be positive")
        if separation_distance < 0:
            raise ValueError("separation_distance must be non-negative")
        if collision_radius < 0:
            raise ValueError("collision_radius must be non-negative")
        if max_speed < 0:
            raise ValueError("max_speed must be non-negative")
        if max_acceleration < 0:
            raise ValueError("max_acceleration must be non-negative")
        if boundary_margin < 0:
            raise ValueError("boundary_margin must be non-negative")
        if battery_drain_per_tick < 0:
            raise ValueError("battery_drain_per_tick must be non-negative")
        if activation not in {"staged", "random"}:
            raise ValueError("activation must be either 'staged' or 'random'")

        self.tick = 0
        self.episode_id = episode_id or f"episode-{uuid4().hex}"
        self.bounds = as_bounds3(bounds)
        self.dt = float(dt)
        self.perception_radius = float(perception_radius)
        self.separation_distance = float(separation_distance)
        self.collision_radius = float(collision_radius)
        self.max_speed = float(max_speed)
        self.max_acceleration = float(max_acceleration)
        self.torus = bool(torus)
        self.boundary_mode: BoundaryMode = "wrap" if torus else boundary_mode
        self.boundary_margin = float(boundary_margin)
        self.target_position = (
            as_vector3(target_position, name="target_position") if target_position is not None else None
        )
        self.policy = policy or BoidsPolicy()
        self.activation = activation
        self.battery_drain_per_tick = float(battery_drain_per_tick)
        self.reward_function: RewardFunction = reward_function or _no_reward
        self.done_function: DoneFunction = done_function or _never_done
        self.transition_log: list[Transition] = []
        self.collect_data = collect_data
        self.np_random = np.random.default_rng(seed)

        self.space = ContinuousSpace(
            dimensions=self.bounds,
            torus=self.torus,
            random=self.random,
            n_agents=max(n_drones, 1),
        )

        position_sampler = initial_position_sampler or _default_position_sampler
        velocity_sampler = initial_velocity_sampler or _default_velocity_sampler
        for _ in range(n_drones):
            position = position_sampler(self.np_random, self.bounds)
            velocity = clip_norm(velocity_sampler(self.np_random, self.max_speed), self.max_speed)
            DroneAgent(
                model=self,
                position=position,
                velocity=velocity,
                policy=self.policy,
                battery=battery_initial,
            )

        self.datacollector = DataCollector(
            model_reporters={
                "tick": lambda model: model.tick,
                "n_drones": lambda model: len(model.agents),
                "transition_count": lambda model: len(model.transition_log),
                "mean_speed": lambda model: mean_speed(model.agents),
                "min_pairwise_distance": lambda model: min_pairwise_distance(model.agents),
                "collision_count": lambda model: collision_count(
                    model.agents,
                    collision_radius=model.collision_radius,
                ),
                "centroid": lambda model: swarm_centroid(model.agents),
            },
            agent_reporters={
                "x": lambda agent: float(agent.position[0]),
                "y": lambda agent: float(agent.position[1]),
                "z": lambda agent: float(agent.position[2]),
                "vx": lambda agent: float(agent.velocity[0]),
                "vy": lambda agent: float(agent.velocity[1]),
                "vz": lambda agent: float(agent.velocity[2]),
                "speed": lambda agent: float(np.linalg.norm(agent.velocity)),
                "mode": lambda agent: agent.mode,
                "battery": lambda agent: agent.battery,
            },
        )

    def step(self) -> None:
        """Advance the simulation by one tick and log per-agent transitions."""

        if self.activation == "staged":
            self.agents.do("cache_state")
            self.agents.do("select_action")
            self.agents.do("propose_motion")
            self.agents.do("commit_motion")
            self.agents.do("log_transition")
        else:
            self.agents.shuffle_do("step_random")

        if self.collect_data:
            self.datacollector.collect(self)

        self.tick += 1

    def run_steps(self, steps: int) -> None:
        """Run the model for *steps* ticks."""

        if steps < 0:
            raise ValueError("steps must be non-negative")
        for _ in range(steps):
            self.step()

    def transitions_dataframe(self):  # type: ignore[no-untyped-def]
        """Return the transition log as a pandas DataFrame."""

        from drone_swarm.io import transitions_to_dataframe

        return transitions_to_dataframe(self.transition_log)

    def export_transitions_csv(self, path: str) -> None:
        """Write flattened transition rows to CSV."""

        write_transitions_csv(self.transition_log, path)

    def export_transitions_jsonl(self, path: str) -> None:
        """Write nested transition records to JSON Lines."""

        write_transitions_jsonl(self.transition_log, path)

    def export_transitions_parquet(self, path: str) -> None:
        """Write flattened transition rows to Parquet."""

        write_transitions_parquet(self.transition_log, path)

    def reset_transition_log(self) -> None:
        """Clear the transition log without moving agents."""

        self.transition_log.clear()
