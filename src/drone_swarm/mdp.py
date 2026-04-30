"""MDP schemas and flattening helpers for drone swarm transitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, TypeAlias

Vector3: TypeAlias = tuple[float, float, float]
ActionType: TypeAlias = Literal[
    "boids_steer",
    "hold",
    "avoid_boundary",
    "avoid_neighbor",
    "return_to_base",
    "random_acceleration",
]


def vector3(value: object, *, name: str = "vector") -> Vector3:
    """Return *value* as a validated 3-tuple of floats.

    The simulator stores positions, velocities, and accelerations as NumPy arrays
    internally, but transition logs should be stable, JSON-friendly Python data.
    """

    try:
        items = tuple(float(x) for x in value)  # type: ignore[arg-type]
    except TypeError as exc:  # pragma: no cover - defensive branch
        raise TypeError(f"{name} must be an iterable of three numeric values") from exc

    if len(items) != 3:
        raise ValueError(f"{name} must contain exactly three values; got {len(items)}")
    return items  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class DroneState:
    """Observable per-agent Markov state at one simulation tick."""

    position: Vector3
    velocity: Vector3
    speed: float
    neighbor_count: int
    nearest_neighbor_distance: float | None
    local_centroid: Vector3 | None
    local_average_velocity: Vector3 | None
    target_vector: Vector3 | None = None
    battery: float | None = None
    mode: str = "nominal"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class DroneAction:
    """Action selected by a drone policy.

    For boids-style control the action is continuous: a 3D acceleration vector.
    ``raw_acceleration`` stores the unclipped policy output when clipping occurs.
    ``components`` can store interpretable steering terms such as cohesion,
    alignment, separation, goal seeking, and boundary avoidance.
    """

    acceleration: Vector3
    action_type: ActionType = "boids_steer"
    clipped: bool = False
    raw_acceleration: Vector3 | None = None
    components: Mapping[str, Vector3] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class Transition:
    """One per-agent MDP transition: ``S_t, A_t, S_{t+1}``."""

    episode_id: str
    step: int
    agent_id: int
    state: DroneState
    action: DroneAction
    next_state: DroneState
    reward: float | None = None
    done: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a nested JSON-serializable dictionary representation."""

        return asdict(self)


def _add_vector_fields(row: dict[str, Any], prefix: str, value: Vector3 | None) -> None:
    """Add vector fields named ``{prefix}_x/y/z`` to *row*."""

    if value is None:
        row[f"{prefix}_x"] = None
        row[f"{prefix}_y"] = None
        row[f"{prefix}_z"] = None
        return

    row[f"{prefix}_x"] = value[0]
    row[f"{prefix}_y"] = value[1]
    row[f"{prefix}_z"] = value[2]


def transition_to_row(transition: Transition) -> dict[str, Any]:
    """Flatten a transition into analysis-friendly scalar columns."""

    row: dict[str, Any] = {
        "episode_id": transition.episode_id,
        "step": transition.step,
        "agent_id": transition.agent_id,
        "reward": transition.reward,
        "done": transition.done,
        "s_speed": transition.state.speed,
        "s_neighbor_count": transition.state.neighbor_count,
        "s_nearest_neighbor_distance": transition.state.nearest_neighbor_distance,
        "s_battery": transition.state.battery,
        "s_mode": transition.state.mode,
        "action_type": transition.action.action_type,
        "action_clipped": transition.action.clipped,
        "sp_speed": transition.next_state.speed,
        "sp_neighbor_count": transition.next_state.neighbor_count,
        "sp_nearest_neighbor_distance": transition.next_state.nearest_neighbor_distance,
        "sp_battery": transition.next_state.battery,
        "sp_mode": transition.next_state.mode,
    }

    _add_vector_fields(row, "s_position", transition.state.position)
    _add_vector_fields(row, "s_velocity", transition.state.velocity)
    _add_vector_fields(row, "s_local_centroid", transition.state.local_centroid)
    _add_vector_fields(row, "s_local_average_velocity", transition.state.local_average_velocity)
    _add_vector_fields(row, "s_target_vector", transition.state.target_vector)

    _add_vector_fields(row, "a_acceleration", transition.action.acceleration)
    _add_vector_fields(row, "a_raw_acceleration", transition.action.raw_acceleration)

    _add_vector_fields(row, "sp_position", transition.next_state.position)
    _add_vector_fields(row, "sp_velocity", transition.next_state.velocity)
    _add_vector_fields(row, "sp_local_centroid", transition.next_state.local_centroid)
    _add_vector_fields(row, "sp_local_average_velocity", transition.next_state.local_average_velocity)
    _add_vector_fields(row, "sp_target_vector", transition.next_state.target_vector)

    if transition.action.components:
        for component_name, component_vector in transition.action.components.items():
            safe_name = component_name.replace(" ", "_").lower()
            _add_vector_fields(row, f"a_component_{safe_name}", component_vector)

    for key, value in transition.info.items():
        row[f"info_{key}"] = value

    return row
