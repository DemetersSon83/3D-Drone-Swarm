"""Mesa model for a 3D boids-style drone swarm."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from mesa import Model  # type: ignore[import-untyped]
from mesa.datacollection import DataCollector  # type: ignore[import-untyped]
from mesa.experimental.continuous_space import ContinuousSpace  # type: ignore[import-untyped]
from numpy.typing import ArrayLike, NDArray

from drone_swarm.drone import DroneAgent
from drone_swarm.events import EventManager, stable_seed
from drone_swarm.io import write_transitions_csv, write_transitions_jsonl, write_transitions_parquet
from drone_swarm.mdp import DroneAction, DroneState, Transition
from drone_swarm.metrics import (
    collision_count,
    mean_speed,
    min_pairwise_distance,
    swarm_centroid,
    swarm_metrics_snapshot,
)
from drone_swarm.physics import BoundaryMode, as_bounds3, as_vector3, clip_norm
from drone_swarm.policies import ActionPolicy, BoidsPolicy

type ActivationMode = Literal["staged", "random"]
type RewardFunction = Callable[[DroneState, DroneAction, DroneState, DroneAgent], float | None]
type DoneFunction = Callable[[DroneState, DroneAction, DroneState, DroneAgent], bool]
type PositionSampler = Callable[[np.random.Generator, NDArray[np.float64]], NDArray[np.float64]]
type VelocitySampler = Callable[[np.random.Generator, float], NDArray[np.float64]]
type TransitionCallback = Callable[[Transition], None]


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


def _materialize_base_seed(seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    generated = np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0]
    return int(generated)


class DroneSwarmModel(Model):  # type: ignore[misc]
    """A 3D Mesa model with reproducible perturbation hooks and dual-view logs.

    The default ``staged`` mode retains move-all semantics.  Initialization,
    policy and perturbation randomness use independent streams so a paired
    nominal/event experiment can share an identical pre-event trajectory.
    """

    def __init__(
        self,
        *,
        n_drones: int = 50,
        bounds: ArrayLike = ((0.0, 100.0), (0.0, 100.0), (0.0, 50.0)),
        seed: int | None = None,
        initialization_seed: int | None = None,
        policy_seed: int | None = None,
        perturbation_seed: int | None = None,
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
        event_manager: EventManager | None = None,
        retain_transitions: bool = True,
        transition_callback: TransitionCallback | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        base_seed = _materialize_base_seed(seed)
        if base_seed < 0:
            raise ValueError("seed must be non-negative")
        super().__init__(rng=base_seed)

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
        self.base_seed = base_seed
        self.initialization_seed = (
            int(initialization_seed)
            if initialization_seed is not None
            else stable_seed(base_seed, "initialization")
        )
        self.policy_seed = (
            int(policy_seed) if policy_seed is not None else stable_seed(base_seed, "policy")
        )
        self.perturbation_seed = (
            int(perturbation_seed)
            if perturbation_seed is not None
            else stable_seed(base_seed, "perturbation")
        )
        self.initialization_rng = np.random.default_rng(self.initialization_seed)
        self.policy_rng = np.random.default_rng(self.policy_seed)
        self.perturbation_rng = np.random.default_rng(self.perturbation_seed)
        # Backward-compatible alias used by external policies.
        self.np_random = self.policy_rng

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
            as_vector3(target_position, name="target_position")
            if target_position is not None
            else None
        )
        self.policy = policy or BoidsPolicy()
        self.activation = activation
        self.battery_drain_per_tick = float(battery_drain_per_tick)
        self.reward_function: RewardFunction = reward_function or _no_reward
        self.done_function: DoneFunction = done_function or _never_done
        self.transition_log: list[Transition] = []
        self.transition_count = 0
        self.retain_transitions = bool(retain_transitions)
        self.transition_callback = transition_callback
        self.collect_data = collect_data
        self.metadata = dict(metadata or {})
        self._last_observed_neighbor_ids: dict[int, tuple[int, ...]] = {}

        self.event_manager = event_manager or EventManager(perturbation_seed=self.perturbation_seed)
        # Respect the model's explicit perturbation seed when callers construct a
        # blank/default manager themselves.
        self.event_manager.perturbation_seed = self.perturbation_seed

        self.space = ContinuousSpace(
            dimensions=self.bounds,
            torus=self.torus,
            random=self.random,
            n_agents=max(n_drones, 1),
        )

        position_sampler = initial_position_sampler or _default_position_sampler
        velocity_sampler = initial_velocity_sampler or _default_velocity_sampler
        for agent_index in range(n_drones):
            position = position_sampler(self.initialization_rng, self.bounds)
            velocity = clip_norm(
                velocity_sampler(self.initialization_rng, self.max_speed),
                self.max_speed,
            )
            DroneAgent(
                model=self,
                position=position,
                velocity=velocity,
                policy=self.policy,
                agent_index=agent_index,
                battery=battery_initial,
                target_position=self.target_position,
                target_id="global" if self.target_position is not None else None,
            )

        self.event_manager.bind(self)

        self.datacollector = DataCollector(
            model_reporters={
                "tick": lambda model: model.tick,
                "n_drones": lambda model: len(model.agents),
                "transition_count": lambda model: model.transition_count,
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
                "coalition_id": lambda agent: agent.coalition_id,
                "role": lambda agent: agent.role,
                "formation_id": lambda agent: agent.formation_id,
            },
        )

    def get_true_neighbors(
        self,
        agent: DroneAgent,
    ) -> tuple[list[DroneAgent], list[float]]:
        """Return the nominal-radius physical neighborhood without event filters."""

        neighbors, distances = agent.get_neighbors_in_radius(self.perception_radius)
        pairs = sorted(
            zip(neighbors, distances, strict=False),
            key=lambda pair: int(pair[0].unique_id),
        )
        return [pair[0] for pair in pairs], [float(pair[1]) for pair in pairs]

    def get_perceived_neighbors(
        self,
        agent: DroneAgent,
    ) -> tuple[list[DroneAgent], list[float]]:
        """Return the realized controller neighborhood after communication events."""

        radius = self.event_manager.effective_perception_radius(self, agent)
        neighbors, distances = agent.get_neighbors_in_radius(radius)
        filtered_neighbors, filtered_distances = self.event_manager.filter_neighbors(
            self,
            agent,
            list(neighbors),
            [float(distance) for distance in distances],
        )
        pairs = sorted(
            zip(filtered_neighbors, filtered_distances, strict=False),
            key=lambda pair: int(pair[0].unique_id),
        )
        return [pair[0] for pair in pairs], [float(pair[1]) for pair in pairs]

    def record_transition(self, transition: Transition) -> None:
        """Record or stream one transition and update realized graph state."""

        self.transition_count += 1
        self._last_observed_neighbor_ids[transition.agent_id] = transition.next_state.neighbor_ids
        if self.retain_transitions:
            self.transition_log.append(transition)
        if self.transition_callback is not None:
            self.transition_callback(transition)

    def step(self) -> None:
        """Advance the simulation by one tick and log per-agent transitions."""

        self.event_manager.before_step(self)
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

        self.event_manager.after_step(self)
        self.tick += 1

    def run_steps(self, steps: int) -> None:
        """Run the model for *steps* ticks."""

        if steps < 0:
            raise ValueError("steps must be non-negative")
        for _ in range(steps):
            self.step()

    def metrics_snapshot(self) -> dict[str, Any]:
        """Return conventional swarm diagnostics for the current physical state."""

        return swarm_metrics_snapshot(
            self.agents,
            tick=self.tick,
            transition_count=self.transition_count,
            collision_radius=self.collision_radius,
            realized_neighbor_ids=self._last_observed_neighbor_ids,
        )

    def agent_registry(self) -> list[dict[str, Any]]:
        """Return stable agent identity and current ground-truth metadata."""

        rows: list[dict[str, Any]] = []
        for agent in sorted(self.agents, key=lambda value: int(value.agent_index)):
            rows.append(
                {
                    "agent_id": int(agent.unique_id),
                    "agent_index": int(agent.agent_index),
                    "coalition_id": agent.coalition_id,
                    "role": agent.role,
                    "formation_id": agent.formation_id,
                    "target_id": agent.target_id,
                    "target_x": None
                    if agent.target_position is None
                    else float(agent.target_position[0]),
                    "target_y": None
                    if agent.target_position is None
                    else float(agent.target_position[1]),
                    "target_z": None
                    if agent.target_position is None
                    else float(agent.target_position[2]),
                    "restrict_interactions_to_coalition": agent.restrict_interactions_to_coalition,
                }
            )
        return rows

    def transitions_dataframe(self) -> Any:
        """Return retained transition logs as a pandas DataFrame."""

        from drone_swarm.io import transitions_to_dataframe

        return transitions_to_dataframe(self.transition_log)

    def export_transitions_csv(self, path: str) -> None:
        """Write retained flattened transition rows to CSV."""

        write_transitions_csv(self.transition_log, path)

    def export_transitions_jsonl(self, path: str) -> None:
        """Write retained nested transition records to JSON Lines."""

        write_transitions_jsonl(self.transition_log, path)

    def export_transitions_parquet(self, path: str) -> None:
        """Write retained flattened transition rows to Parquet."""

        write_transitions_parquet(self.transition_log, path)

    def reset_transition_log(self) -> None:
        """Clear the transition log and counters without moving agents."""

        self.transition_log.clear()
        self.transition_count = 0
        self._last_observed_neighbor_ids.clear()
