"""MDP schemas and flattening helpers for drone swarm transitions."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

type Vector3 = tuple[float, float, float]
type AgentIds = tuple[int, ...]
type ActionType = Literal[
    "boids_steer",
    "hold",
    "avoid_boundary",
    "avoid_neighbor",
    "return_to_base",
    "random_acceleration",
]

_STANDARD_COMPONENTS = ("cohesion", "alignment", "separation", "goal", "boundary")


def vector3(value: object, *, name: str = "vector") -> Vector3:
    """Return *value* as a validated 3-tuple of finite floats."""

    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of three numeric values")
    try:
        items = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
        raise TypeError(f"{name} must be an iterable of three numeric values") from exc

    if len(items) != 3:
        raise ValueError(f"{name} must contain exactly three values; got {len(items)}")
    if not all(item == item and abs(item) != float("inf") for item in items):
        raise ValueError(f"{name} must contain only finite values")
    return (items[0], items[1], items[2])


@dataclass(frozen=True, slots=True)
class DroneState:
    """Observable per-agent Markov state at one simulation tick.

    ``neighbor_ids`` and ``local_separation`` make the logged state sufficient for
    the built-in boids controller.  This is important for information-theoretic
    analysis: the action should be a function of the state that is actually
    recorded, rather than of hidden neighbor objects queried by the policy.
    """

    position: Vector3
    velocity: Vector3
    speed: float
    neighbor_count: int
    nearest_neighbor_distance: float | None
    local_centroid: Vector3 | None
    local_average_velocity: Vector3 | None
    neighbor_ids: AgentIds = ()
    local_separation: Vector3 | None = None
    target_vector: Vector3 | None = None
    battery: float | None = None
    mode: str = "nominal"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class DroneAction:
    """Action selected by a drone policy.

    ``acceleration`` is the command at the relevant boundary.  A transition can
    store both the commanded and applied actions, so actuator faults can be
    distinguished from policy changes.  ``metadata`` is intentionally small and
    JSON-compatible; event truth belongs on the transition and in ``events``.
    """

    acceleration: Vector3
    action_type: ActionType = "boids_steer"
    clipped: bool = False
    raw_acceleration: Vector3 | None = None
    components: Mapping[str, Vector3] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class Transition:
    """One per-agent transition with controller- and plant-facing views.

    Backward-compatible fields ``state``, ``action`` and ``next_state`` are the
    controller-facing observed state, commanded action, and observed outcome.
    The optional true/applied fields provide a plant-facing interaction token:
    ``(true_state, applied_action, true_next_state)``.
    """

    episode_id: str
    step: int
    agent_id: int
    state: DroneState
    action: DroneAction
    next_state: DroneState
    reward: float | None = None
    done: bool = False
    true_state: DroneState | None = None
    applied_action: DroneAction | None = None
    environment_acceleration: Vector3 | None = None
    true_next_state: DroneState | None = None
    phase: str | None = None
    active_event_ids: tuple[str, ...] = ()
    agent_affected: bool = False
    coalition_truth: str | None = None
    role_truth: str | None = None
    formation_truth: str | None = None
    target_id: str | None = None
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


def _add_state_fields(row: dict[str, Any], prefix: str, state: DroneState | None) -> None:
    if state is None:
        row[f"{prefix}_speed"] = None
        row[f"{prefix}_neighbor_count"] = None
        row[f"{prefix}_nearest_neighbor_distance"] = None
        row[f"{prefix}_neighbor_ids"] = "[]"
        row[f"{prefix}_battery"] = None
        row[f"{prefix}_mode"] = None
        for name in (
            "position",
            "velocity",
            "local_centroid",
            "local_average_velocity",
            "local_separation",
            "target_vector",
        ):
            _add_vector_fields(row, f"{prefix}_{name}", None)
        return

    row[f"{prefix}_speed"] = state.speed
    row[f"{prefix}_neighbor_count"] = state.neighbor_count
    row[f"{prefix}_nearest_neighbor_distance"] = state.nearest_neighbor_distance
    row[f"{prefix}_neighbor_ids"] = json.dumps(state.neighbor_ids, separators=(",", ":"))
    row[f"{prefix}_battery"] = state.battery
    row[f"{prefix}_mode"] = state.mode
    _add_vector_fields(row, f"{prefix}_position", state.position)
    _add_vector_fields(row, f"{prefix}_velocity", state.velocity)
    _add_vector_fields(row, f"{prefix}_local_centroid", state.local_centroid)
    _add_vector_fields(row, f"{prefix}_local_average_velocity", state.local_average_velocity)
    _add_vector_fields(row, f"{prefix}_local_separation", state.local_separation)
    _add_vector_fields(row, f"{prefix}_target_vector", state.target_vector)


def _add_action_fields(row: dict[str, Any], prefix: str, action: DroneAction | None) -> None:
    row[f"{prefix}_type"] = action.action_type if action is not None else None
    row[f"{prefix}_clipped"] = action.clipped if action is not None else None
    row[f"{prefix}_metadata_json"] = (
        json.dumps(action.metadata, sort_keys=True, separators=(",", ":"), default=str)
        if action is not None
        else "{}"
    )
    _add_vector_fields(row, f"{prefix}_acceleration", action.acceleration if action else None)
    _add_vector_fields(
        row,
        f"{prefix}_raw_acceleration",
        action.raw_acceleration if action else None,
    )

    components = action.components if action is not None and action.components else {}
    row[f"{prefix}_components_json"] = json.dumps(
        components,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    for component_name in _STANDARD_COMPONENTS:
        _add_vector_fields(
            row,
            f"{prefix}_component_{component_name}",
            components.get(component_name),
        )

    # Preserve custom components while keeping the built-in schema stable.
    for component_name, component_vector in components.items():
        safe_name = component_name.replace(" ", "_").lower()
        if component_name not in _STANDARD_COMPONENTS:
            _add_vector_fields(row, f"{prefix}_component_{safe_name}", component_vector)


def transition_to_row(transition: Transition) -> dict[str, Any]:
    """Flatten a transition into analysis-friendly scalar columns.

    Prefixes are deliberately explicit:

    - ``s_*``, ``a_*``, ``sp_*``: observed state, command, observed outcome.
    - ``true_s_*``, ``applied_a_*``, ``true_sp_*``: plant-facing token.
    - ``environment_acceleration_*``: exogenous acceleration added after the
      actuator boundary.
    """

    row: dict[str, Any] = {
        "run_id": transition.episode_id,
        "episode_id": transition.episode_id,
        "step": transition.step,
        "agent_id": transition.agent_id,
        "reward": transition.reward,
        "done": transition.done,
        "phase": transition.phase,
        "active_event_ids": json.dumps(transition.active_event_ids, separators=(",", ":")),
        "agent_affected": transition.agent_affected,
        "coalition_truth": transition.coalition_truth,
        "role_truth": transition.role_truth,
        "formation_truth": transition.formation_truth,
        "target_id": transition.target_id,
        "info_json": json.dumps(
            transition.info,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
    }

    _add_state_fields(row, "s", transition.state)
    _add_action_fields(row, "a", transition.action)
    # Backward-compatible aliases retained from the original flat schema.
    row["action_type"] = transition.action.action_type
    row["action_clipped"] = transition.action.clipped
    _add_state_fields(row, "sp", transition.next_state)
    _add_state_fields(row, "true_s", transition.true_state)
    _add_action_fields(row, "applied_a", transition.applied_action)
    _add_vector_fields(row, "environment_acceleration", transition.environment_acceleration)
    _add_state_fields(row, "true_sp", transition.true_next_state)

    # Retain the original convenience behavior for scalar ``info`` values.
    for key, value in transition.info.items():
        if value is None or isinstance(value, (bool, int, float, str)):
            row[f"info_{key}"] = value

    return row
