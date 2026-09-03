"""Scenario configuration, validation, and model construction helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from drone_swarm.events import EventManager, resolve_assignments, stable_seed
from drone_swarm.mdp import Transition
from drone_swarm.physics import BoundaryMode, as_vector3, clip_norm, unit_vector
from drone_swarm.policies import ActionPolicy, BoidsPolicy, HoldPolicy, RandomAccelerationPolicy

if TYPE_CHECKING:
    from drone_swarm.model import DroneSwarmModel


class ScenarioError(ValueError):
    """Raised when a scenario file is malformed or internally inconsistent."""


_MAX_UINT64 = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class RunSeeds:
    base_seed: int
    initialization_seed: int
    policy_seed: int
    perturbation_seed: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def resolve_run_seeds(
    base_seed: int,
    *,
    initialization_seed: int | None = None,
    policy_seed: int | None = None,
    perturbation_seed: int | None = None,
) -> RunSeeds:
    """Resolve independent RNG streams from one paired-run seed."""

    seed_values = {
        "base_seed": int(base_seed),
        "initialization_seed": (
            int(initialization_seed)
            if initialization_seed is not None
            else stable_seed(base_seed, "initialization")
        ),
        "policy_seed": (
            int(policy_seed) if policy_seed is not None else stable_seed(base_seed, "policy")
        ),
        "perturbation_seed": (
            int(perturbation_seed)
            if perturbation_seed is not None
            else stable_seed(base_seed, "perturbation")
        ),
    }
    negative = {name: value for name, value in seed_values.items() if value < 0}
    if negative:
        raise ValueError(f"seeds must be non-negative: {negative}")
    too_large = {name: value for name, value in seed_values.items() if value > _MAX_UINT64}
    if too_large:
        raise ValueError(f"seeds must fit in unsigned 64-bit integers: {too_large}")
    return RunSeeds(**seed_values)


def load_scenario(path: str | Path) -> dict[str, Any]:
    """Load YAML or JSON and return a validated mutable mapping."""

    scenario_path = Path(path)
    if not scenario_path.is_file():
        raise FileNotFoundError(f"scenario file not found: {scenario_path}")
    text = scenario_path.read_text(encoding="utf-8")
    if scenario_path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - optional at runtime
            raise RuntimeError(
                "YAML scenarios require PyYAML. Install with: python -m pip install PyYAML"
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise ScenarioError("scenario root must be a mapping")
    config = dict(value)
    validate_scenario(config)
    return config


def validate_scenario(config: Mapping[str, Any]) -> None:
    schema_version = int(config.get("schema_version", 1))
    if schema_version != 1:
        raise ScenarioError(f"unsupported scenario schema_version: {schema_version}")
    scenario_id = str(config.get("scenario_id", "")).strip()
    if not scenario_id:
        raise ScenarioError("scenario_id is required")

    run = config.get("run", {})
    model = config.get("model", {})
    policy = config.get("policy", {})
    output = config.get("output", {})
    initialization = config.get("initialization", {})
    agents = config.get("agents", {})
    named_sections = {
        "run": run,
        "model": model,
        "policy": policy,
        "output": output,
        "initialization": initialization,
        "agents": agents,
    }
    invalid_sections = [
        name for name, value in named_sections.items() if not isinstance(value, Mapping)
    ]
    if invalid_sections:
        raise ScenarioError(f"scenario sections must be mappings: {invalid_sections}")

    steps = int(run.get("steps", 0))
    if steps <= 0:
        raise ScenarioError("run.steps must be positive")
    if int(run.get("metrics_stride", 1)) <= 0:
        raise ScenarioError("run.metrics_stride must be positive")
    if int(run.get("transition_batch_size", 20_000)) <= 0:
        raise ScenarioError("run.transition_batch_size must be positive")

    n_drones = int(model.get("n_drones", 48))
    if n_drones < 0:
        raise ScenarioError("model.n_drones must be non-negative")
    if float(model.get("dt", 0.25)) <= 0:
        raise ScenarioError("model.dt must be positive")
    if float(model.get("perception_radius", 14.0)) <= 0:
        raise ScenarioError("model.perception_radius must be positive")
    if float(model.get("separation_distance", 2.5)) < 0:
        raise ScenarioError("model.separation_distance must be non-negative")
    if float(model.get("collision_radius", 0.75)) < 0:
        raise ScenarioError("model.collision_radius must be non-negative")
    if float(model.get("max_speed", 3.0)) < 0:
        raise ScenarioError("model.max_speed must be non-negative")
    if float(model.get("max_acceleration", 0.5)) < 0:
        raise ScenarioError("model.max_acceleration must be non-negative")
    if float(model.get("boundary_margin", 5.0)) < 0:
        raise ScenarioError("model.boundary_margin must be non-negative")
    boundary_mode = str(model.get("boundary_mode", "bounce"))
    if boundary_mode not in {"clip", "bounce", "wrap"}:
        raise ScenarioError(f"unsupported model.boundary_mode: {boundary_mode}")
    activation = str(model.get("activation", "staged"))
    if activation not in {"staged", "random"}:
        raise ScenarioError(f"unsupported model.activation: {activation}")

    policy_type = str(policy.get("type", "boids"))
    if policy_type not in {"boids", "hold", "random_acceleration"}:
        raise ScenarioError(f"unsupported policy.type: {policy_type}")

    formats = output.get("formats", ["parquet"])
    if isinstance(formats, str):
        formats = [formats]
    if not isinstance(formats, Sequence):
        raise ScenarioError("output.formats must be a string or sequence")
    normalized_formats = [str(value).lower() for value in formats]
    unsupported_formats = set(normalized_formats).difference({"parquet", "csv", "jsonl"})
    if unsupported_formats:
        raise ScenarioError(f"unsupported output formats: {sorted(unsupported_formats)}")
    if not normalized_formats:
        raise ScenarioError("output.formats must contain at least one format")

    event_values = config.get("events", ())
    if not isinstance(event_values, Sequence) or isinstance(event_values, (str, bytes)):
        raise ScenarioError("events must be a sequence of mappings")
    events: list[Mapping[str, Any]] = []
    for event in event_values:
        if not isinstance(event, Mapping):
            raise ScenarioError("each event must be a mapping")
        events.append(event)
        event_id = str(event.get("id", event.get("event_id", "")))
        schedule = event.get("schedule", {})
        if not isinstance(schedule, Mapping):
            raise ScenarioError(f"event {event_id or '<unnamed>'} schedule must be a mapping")
        start_step = int(schedule.get("start", schedule.get("start_step", 0)))
        if start_step >= steps:
            raise ScenarioError(f"event {event_id or '<unnamed>'} starts after run.steps")

    phase_values = config.get("phases", ())
    if not isinstance(phase_values, Sequence) or isinstance(phase_values, (str, bytes)):
        raise ScenarioError("phases must be a sequence of mappings")
    phases: list[Mapping[str, Any]] = []
    for phase in phase_values:
        if not isinstance(phase, Mapping) or not phase.get("name"):
            raise ScenarioError("each phase requires a name")
        phases.append(phase)

    try:
        EventManager.from_config(events, perturbation_seed=0, phases=phases)
        make_policy(policy)
        position_config = initialization.get("positions")
        velocity_config = initialization.get("velocities")
        make_position_sampler(position_config if isinstance(position_config, Mapping) else None)
        make_velocity_sampler(velocity_config if isinstance(velocity_config, Mapping) else None)
    except (TypeError, ValueError) as exc:
        raise ScenarioError(str(exc)) from exc


def canonical_config_json(config: Mapping[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def scenario_config_hash(config: Mapping[str, Any]) -> str:
    """Hash the complete resolved scenario, including output/runtime settings."""

    return hashlib.sha256(canonical_config_json(config).encode()).hexdigest()


def simulation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields that can affect simulated trajectories or labels.

    Output formats, batching, progress logging, analysis splits, and pairing
    metadata are deliberately excluded. This keeps a run identity stable when
    the same scientific run is exported in a different artifact format.
    """

    run = config.get("run", {})
    steps = int(run.get("steps", 0)) if isinstance(run, Mapping) else 0
    return {
        "schema_version": int(config.get("schema_version", 1)),
        "scenario_id": str(config.get("scenario_id", "")),
        "run": {"steps": steps},
        "model": config.get("model", {}),
        "policy": config.get("policy", {}),
        "initialization": config.get("initialization", {}),
        "agents": config.get("agents", {}),
        "phases": config.get("phases", ()),
        "events": config.get("events", ()),
    }


def simulation_config_hash(config: Mapping[str, Any]) -> str:
    """Hash the scientific simulation definition used in deterministic IDs."""

    return hashlib.sha256(canonical_config_json(simulation_config(config)).encode()).hexdigest()


def sanitize_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return cleaned or "scenario"


def make_run_id(config: Mapping[str, Any], seeds: RunSeeds) -> str:
    scenario_id = sanitize_identifier(str(config["scenario_id"]))
    payload = {
        "simulation_hash": simulation_config_hash(config),
        "seeds": seeds.to_dict(),
    }
    run_hash = hashlib.sha256(canonical_config_json(payload).encode()).hexdigest()[:10]
    return f"{scenario_id}__seed-{seeds.base_seed:06d}__{run_hash}"


def make_policy(config: Mapping[str, Any]) -> ActionPolicy:
    policy_type = str(config.get("type", "boids"))
    if policy_type == "hold":
        return HoldPolicy()
    if policy_type == "random_acceleration":
        return RandomAccelerationPolicy(scale=float(config.get("scale", 1.0)))
    return BoidsPolicy(
        cohesion_weight=float(config.get("cohesion_weight", 0.025)),
        alignment_weight=float(config.get("alignment_weight", 0.05)),
        separation_weight=float(config.get("separation_weight", 0.25)),
        goal_weight=float(config.get("goal_weight", 0.02)),
        boundary_weight=float(config.get("boundary_weight", 1.0)),
    )


def make_position_sampler(
    config: Mapping[str, Any] | None,
) -> Callable[[np.random.Generator, NDArray[np.float64]], NDArray[np.float64]] | None:
    if not config or str(config.get("type", "uniform")) == "uniform":
        return None
    sampler_type = str(config.get("type"))

    if sampler_type == "gaussian":
        center_value = config.get("center")
        std = np.asarray(config.get("std", 5.0), dtype=float)
        if std.ndim == 0:
            std = np.full(3, float(std))
        std = as_vector3(std, name="position std")

        def sample_gaussian(
            rng: np.random.Generator,
            bounds: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            center = (
                bounds.mean(axis=1)
                if center_value is None
                else as_vector3(center_value, name="position center")
            )
            return np.clip(rng.normal(center, std, size=3), bounds[:, 0], bounds[:, 1])

        return sample_gaussian

    if sampler_type == "clusters":
        clusters = config.get("clusters", ())
        if not isinstance(clusters, Sequence) or not clusters:
            raise ScenarioError("clusters position sampler requires a non-empty clusters list")
        centers = [as_vector3(cluster["center"], name="cluster center") for cluster in clusters]
        stds = []
        weights = []
        for cluster in clusters:
            std_value = np.asarray(cluster.get("std", 3.0), dtype=float)
            if std_value.ndim == 0:
                std_value = np.full(3, float(std_value))
            stds.append(as_vector3(std_value, name="cluster std"))
            weights.append(float(cluster.get("weight", 1.0)))
        probabilities = np.asarray(weights, dtype=float)
        probabilities /= probabilities.sum()

        def sample_cluster(
            rng: np.random.Generator,
            bounds: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            index = int(rng.choice(len(centers), p=probabilities))
            return np.clip(
                rng.normal(centers[index], stds[index], size=3),
                bounds[:, 0],
                bounds[:, 1],
            )

        return sample_cluster

    if sampler_type == "grid":
        shape = tuple(int(value) for value in config.get("shape", (4, 4, 3)))
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise ScenarioError("grid shape must contain three positive integers")
        margin = float(config.get("margin", 0.1))
        points: list[np.ndarray] = []
        counter = 0

        def sample_grid(
            rng: np.random.Generator,
            bounds: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            nonlocal counter, points
            del rng
            if not points:
                low = bounds[:, 0] + margin * (bounds[:, 1] - bounds[:, 0])
                high = bounds[:, 1] - margin * (bounds[:, 1] - bounds[:, 0])
                axes = [np.linspace(low[axis], high[axis], shape[axis]) for axis in range(3)]
                points = [
                    np.asarray([x, y, z], dtype=float)
                    for x in axes[0]
                    for y in axes[1]
                    for z in axes[2]
                ]
            value = points[counter % len(points)].copy()
            counter += 1
            return value

        return sample_grid

    raise ScenarioError(f"unsupported initialization.positions.type: {sampler_type}")


def make_velocity_sampler(
    config: Mapping[str, Any] | None,
) -> Callable[[np.random.Generator, float], NDArray[np.float64]] | None:
    if not config or str(config.get("type", "random")) == "random":
        return None
    sampler_type = str(config.get("type"))

    if sampler_type == "zero":
        return lambda rng, max_speed: np.zeros(3, dtype=float)

    if sampler_type == "aligned":
        direction = unit_vector(as_vector3(config.get("direction", (1.0, 0.0, 0.0))))
        speed_fraction = float(config.get("speed_fraction", 0.5))
        direction_noise = float(config.get("direction_noise", 0.05))

        def sample_aligned(rng: np.random.Generator, max_speed: float) -> NDArray[np.float64]:
            noisy_direction = unit_vector(direction + rng.normal(0.0, direction_noise, size=3))
            return clip_norm(noisy_direction * max_speed * speed_fraction, max_speed)

        return sample_aligned

    if sampler_type == "gaussian":
        mean = as_vector3(config.get("mean", (0.0, 0.0, 0.0)), name="velocity mean")
        std = np.asarray(config.get("std", 0.2), dtype=float)
        if std.ndim == 0:
            std = np.full(3, float(std))
        std = as_vector3(std, name="velocity std")

        def sample_velocity(rng: np.random.Generator, max_speed: float) -> NDArray[np.float64]:
            return clip_norm(rng.normal(mean, std, size=3), max_speed)

        return sample_velocity

    raise ScenarioError(f"unsupported initialization.velocities.type: {sampler_type}")


def _apply_initial_agent_configuration(model: Any, config: Mapping[str, Any]) -> None:
    agents_config = config.get("agents", {})
    if not isinstance(agents_config, Mapping):
        raise ScenarioError("agents must be a mapping")
    agents = list(model.agents)
    all_indices = frozenset(int(agent.agent_index) for agent in agents)

    coalition_spec = agents_config.get("initial_coalitions", {"strategy": "single", "label": "all"})
    if coalition_spec:
        if not isinstance(coalition_spec, Mapping):
            raise ScenarioError("agents.initial_coalitions must be a mapping")
        assignments = resolve_assignments(coalition_spec, agents, all_indices)
        for agent in agents:
            if int(agent.agent_index) in assignments:
                agent.coalition_id = assignments[int(agent.agent_index)]

    if "restrict_interactions_to_coalition" in agents_config:
        restricted = bool(agents_config["restrict_interactions_to_coalition"])
        for agent in agents:
            agent.restrict_interactions_to_coalition = restricted

    if agents_config.get("formation_id") is not None:
        formation_id = str(agents_config["formation_id"])
        for agent in agents:
            agent.formation_id = formation_id

    group_targets = agents_config.get("group_targets", {})
    individual_targets = agents_config.get("agent_targets", {})
    if group_targets and not isinstance(group_targets, Mapping):
        raise ScenarioError("agents.group_targets must be a mapping")
    if individual_targets and not isinstance(individual_targets, Mapping):
        raise ScenarioError("agents.agent_targets must be a mapping")
    for agent in agents:
        index = int(agent.agent_index)
        target_value = None
        if str(index) in individual_targets:
            target_value = individual_targets[str(index)]
        elif index in individual_targets:
            target_value = individual_targets[index]
        elif agent.coalition_id in group_targets:
            target_value = group_targets[agent.coalition_id]
        if target_value is not None:
            agent.target_position = as_vector3(target_value, name="initial target")
            agent.target_id = f"initial:{agent.coalition_id or index}"


def build_model(
    config: Mapping[str, Any],
    *,
    seeds: RunSeeds,
    run_id: str,
    transition_callback: Callable[[Transition], None] | None = None,
    retain_transitions: bool = False,
) -> DroneSwarmModel:
    """Construct a configured model and bind events after initial labels exist."""

    from drone_swarm.model import DroneSwarmModel

    model_config = dict(config.get("model", {}))
    initialization = config.get("initialization", {})
    if not isinstance(initialization, Mapping):
        raise ScenarioError("initialization must be a mapping")

    position_config = initialization.get("positions")
    velocity_config = initialization.get("velocities")
    position_sampler = make_position_sampler(
        position_config if isinstance(position_config, Mapping) else None
    )
    velocity_sampler = make_velocity_sampler(
        velocity_config if isinstance(velocity_config, Mapping) else None
    )

    # Bind a blank manager first. Event selectors that reference planted
    # coalitions are resolved only after initial agent metadata is assigned.
    blank_manager = EventManager(
        perturbation_seed=seeds.perturbation_seed,
        phases=(),
    )
    boundary_mode_value = str(model_config.get("boundary_mode", "bounce"))
    if boundary_mode_value not in {"clip", "bounce", "wrap"}:
        raise ScenarioError(f"unsupported boundary_mode: {boundary_mode_value}")
    boundary_mode = cast(BoundaryMode, boundary_mode_value)

    activation_value = str(model_config.get("activation", "staged"))
    if activation_value not in {"staged", "random"}:
        raise ScenarioError(f"unsupported activation mode: {activation_value}")
    activation = cast(Literal["staged", "random"], activation_value)

    model = DroneSwarmModel(
        n_drones=int(model_config.get("n_drones", 48)),
        bounds=model_config.get("bounds", ((0.0, 60.0), (0.0, 60.0), (0.0, 30.0))),
        seed=seeds.base_seed,
        initialization_seed=seeds.initialization_seed,
        policy_seed=seeds.policy_seed,
        perturbation_seed=seeds.perturbation_seed,
        dt=float(model_config.get("dt", 0.25)),
        perception_radius=float(model_config.get("perception_radius", 14.0)),
        separation_distance=float(model_config.get("separation_distance", 2.5)),
        collision_radius=float(model_config.get("collision_radius", 0.75)),
        max_speed=float(model_config.get("max_speed", 3.0)),
        max_acceleration=float(model_config.get("max_acceleration", 0.5)),
        torus=bool(model_config.get("torus", False)),
        boundary_mode=boundary_mode,
        boundary_margin=float(model_config.get("boundary_margin", 5.0)),
        target_position=model_config.get("target_position"),
        policy=make_policy(config.get("policy", {})),
        activation=activation,
        battery_initial=(
            float(model_config["battery_initial"])
            if model_config.get("battery_initial") is not None
            else None
        ),
        battery_drain_per_tick=float(model_config.get("battery_drain_per_tick", 0.0)),
        initial_position_sampler=position_sampler,
        initial_velocity_sampler=velocity_sampler,
        collect_data=False,
        episode_id=run_id,
        event_manager=blank_manager,
        retain_transitions=retain_transitions,
        transition_callback=transition_callback,
        metadata={
            "scenario_id": config["scenario_id"],
            "analysis_split": config.get("analysis_split", "unspecified"),
        },
    )

    _apply_initial_agent_configuration(model, config)
    manager = EventManager.from_config(
        config.get("events", ()),
        perturbation_seed=seeds.perturbation_seed,
        phases=config.get("phases", ()),
    )
    manager.bind(model)
    model.event_manager = manager
    return model
