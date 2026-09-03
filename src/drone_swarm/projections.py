"""Stable analysis-oriented projections of canonical transition records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from drone_swarm.contracts import AGENT_SIGNALS_CONTRACT_VERSION
from drone_swarm.mdp import DroneAction, DroneState, Transition, Vector3


def _put_vector(row: dict[str, Any], prefix: str, value: Vector3 | None) -> None:
    if value is None:
        row[f"{prefix}_x"] = None
        row[f"{prefix}_y"] = None
        row[f"{prefix}_z"] = None
        return
    row[f"{prefix}_x"] = value[0]
    row[f"{prefix}_y"] = value[1]
    row[f"{prefix}_z"] = value[2]


def _put_state(row: dict[str, Any], prefix: str, state: DroneState | None) -> None:
    if state is None:
        row[f"{prefix}_speed"] = None
        row[f"{prefix}_neighbor_count"] = None
        row[f"{prefix}_nearest_neighbor_distance"] = None
        row[f"{prefix}_neighbor_ids"] = "[]"
        row[f"{prefix}_battery"] = None
        row[f"{prefix}_mode"] = None
        for field in (
            "position",
            "velocity",
            "local_centroid",
            "local_average_velocity",
            "local_separation",
            "target_vector",
        ):
            _put_vector(row, f"{prefix}_{field}", None)
        return

    row[f"{prefix}_speed"] = state.speed
    row[f"{prefix}_neighbor_count"] = state.neighbor_count
    row[f"{prefix}_nearest_neighbor_distance"] = state.nearest_neighbor_distance
    row[f"{prefix}_neighbor_ids"] = json.dumps(state.neighbor_ids, separators=(",", ":"))
    row[f"{prefix}_battery"] = state.battery
    row[f"{prefix}_mode"] = state.mode
    _put_vector(row, f"{prefix}_position", state.position)
    _put_vector(row, f"{prefix}_velocity", state.velocity)
    _put_vector(row, f"{prefix}_local_centroid", state.local_centroid)
    _put_vector(row, f"{prefix}_local_average_velocity", state.local_average_velocity)
    _put_vector(row, f"{prefix}_local_separation", state.local_separation)
    _put_vector(row, f"{prefix}_target_vector", state.target_vector)


def _put_action(row: dict[str, Any], prefix: str, action: DroneAction | None) -> None:
    row[f"{prefix}_type"] = None if action is None else action.action_type
    row[f"{prefix}_clipped"] = None if action is None else action.clipped
    _put_vector(row, f"{prefix}_acceleration", None if action is None else action.acceleration)
    for name in ("cohesion", "alignment", "separation", "goal", "boundary"):
        component = None
        if action is not None and action.components:
            component = action.components.get(name)
        _put_vector(row, f"{prefix}_component_{name}", component)


def agent_signal_row(
    transition: Transition,
    *,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a compact, stable row for common analytical pipelines.

    The canonical `transitions` table remains the source of truth. This
    projection removes low-use fields and gives semantically explicit names so
    analysts can load the controller and plant interaction tokens without
    deciphering legacy prefixes.
    """

    dt = float(context.get("dt", 1.0))
    row: dict[str, Any] = {
        "contract_version": AGENT_SIGNALS_CONTRACT_VERSION,
        "run_id": transition.episode_id,
        "scenario_id": str(context.get("scenario_id", "unknown")),
        "analysis_split": str(context.get("analysis_split", "unspecified")),
        "base_seed": int(context.get("base_seed", 0)),
        "initialization_seed": int(context.get("initialization_seed", 0)),
        "policy_seed": int(context.get("policy_seed", 0)),
        "perturbation_seed": int(context.get("perturbation_seed", 0)),
        "step": transition.step,
        "time": transition.step * dt,
        "next_time": (transition.step + 1) * dt,
        "agent_id": transition.agent_id,
        "agent_index": transition.info.get("agent_index"),
        "phase": transition.phase,
        "active_event_ids": json.dumps(transition.active_event_ids, separators=(",", ":")),
        "agent_affected": transition.agent_affected,
        "coalition_truth": transition.coalition_truth,
        "role_truth": transition.role_truth,
        "formation_truth": transition.formation_truth,
        "target_id": transition.target_id,
        "reward": transition.reward,
        "done": transition.done,
    }

    _put_state(row, "observed_state", transition.state)
    _put_action(row, "commanded_action", transition.action)
    _put_state(row, "observed_next_state", transition.next_state)
    _put_state(row, "true_state", transition.true_state)
    _put_action(row, "applied_action", transition.applied_action)
    _put_vector(row, "environment_acceleration", transition.environment_acceleration)
    _put_state(row, "true_next_state", transition.true_next_state)
    return row


def template_agent_signal_row() -> dict[str, Any]:
    """Return a fully populated row used to freeze projection columns/types."""

    from drone_swarm.mdp import DroneAction, DroneState

    state = DroneState(
        position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        speed=0.0,
        neighbor_count=1,
        nearest_neighbor_distance=1.0,
        local_centroid=(0.0, 0.0, 0.0),
        local_average_velocity=(0.0, 0.0, 0.0),
        neighbor_ids=(1,),
        local_separation=(0.0, 0.0, 0.0),
        target_vector=(0.0, 0.0, 0.0),
        battery=1.0,
        mode="nominal",
    )
    components = {
        name: (0.0, 0.0, 0.0)
        for name in ("cohesion", "alignment", "separation", "goal", "boundary")
    }
    action = DroneAction(
        acceleration=(0.0, 0.0, 0.0),
        raw_acceleration=(0.0, 0.0, 0.0),
        components=components,
    )
    transition = Transition(
        episode_id="template",
        step=0,
        agent_id=1,
        state=state,
        action=action,
        next_state=state,
        reward=0.0,
        done=False,
        true_state=state,
        applied_action=action,
        environment_acceleration=(0.0, 0.0, 0.0),
        true_next_state=state,
        phase="template",
        active_event_ids=("template",),
        agent_affected=True,
        coalition_truth="template",
        role_truth="template",
        formation_truth="template",
        target_id="template",
        info={"agent_index": 0},
    )
    return agent_signal_row(
        transition,
        context={
            "scenario_id": "template",
            "analysis_split": "template",
            "base_seed": 0,
            "initialization_seed": 1,
            "policy_seed": 2,
            "perturbation_seed": 3,
            "dt": 1.0,
        },
    )
