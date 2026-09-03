"""Scheduled perturbations and reconfiguration events for swarm experiments.

The event layer is deliberately deterministic with respect to call order.  Noise
samples and communication-edge drops are generated from stable hashes of the
perturbation seed, event id, tick, agent and sample key.  Adding an unrelated
observer or changing worker scheduling therefore does not silently change the
realized disturbance.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike

from drone_swarm.mdp import DroneAction, DroneState, Vector3, vector3
from drone_swarm.physics import as_vector3, clip_norm

if TYPE_CHECKING:  # pragma: no cover
    from drone_swarm.drone import DroneAgent
    from drone_swarm.model import DroneSwarmModel

type ScheduleShape = Literal["step", "ramp", "pulse", "intermittent"]

SUPPORTED_EVENT_KINDS = frozenset(
    {
        "observation_noise",
        "observation_bias",
        "observation_quantization",
        "perception_radius_scale",
        "communication_dropout",
        "communication_partition",
        "policy_weight_scale",
        "random_action",
        "hold_command",
        "actuator_gain",
        "actuator_noise",
        "actuator_bias",
        "actuator_stuck_axis",
        "actuator_saturation",
        "wind",
        "target_shift",
        "coalition_reconfigure",
    }
)

_EVENT_STAGE = {
    "observation_noise": "observation",
    "observation_bias": "observation",
    "observation_quantization": "observation",
    "perception_radius_scale": "communication",
    "communication_dropout": "communication",
    "communication_partition": "communication",
    "policy_weight_scale": "policy",
    "random_action": "policy",
    "hold_command": "policy",
    "actuator_gain": "actuator",
    "actuator_noise": "actuator",
    "actuator_bias": "actuator",
    "actuator_stuck_axis": "actuator",
    "actuator_saturation": "actuator",
    "wind": "physics",
    "target_shift": "configuration",
    "coalition_reconfigure": "configuration",
}


def stable_seed(base_seed: int, *parts: object) -> int:
    """Return a stable unsigned 64-bit seed derived from *base_seed* and parts."""

    payload = "\x1f".join([str(base_seed), *(str(part) for part in parts)]).encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"swarmevt").digest()
    return int.from_bytes(digest, "little", signed=False)


def _stable_uniform(base_seed: int, *parts: object) -> float:
    # Convert the upper 53 bits to the same precision used by IEEE-754 doubles.
    value = stable_seed(base_seed, *parts) >> 11
    return value / float(1 << 53)


def _stable_normal(
    base_seed: int,
    size: int,
    *parts: object,
) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(base_seed, *parts))
    return np.asarray(rng.normal(size=size), dtype=float)


def _as_float_vector(value: object, *, default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(3, default, dtype=float)
    if isinstance(value, (int, float)):
        return np.full(3, float(value), dtype=float)
    return as_vector3(cast(ArrayLike, value))


def _interpolate(base: float, target: float, intensity: float) -> float:
    return base + intensity * (target - base)


@dataclass(frozen=True, slots=True)
class EventSchedule:
    """A half-open event interval ``[start_step, end_step)`` and intensity rule."""

    start_step: int
    end_step: int | None = None
    shape: ScheduleShape = "step"
    ramp_steps: int | None = None
    period: int | None = None
    duty_cycle: float = 0.5

    def __post_init__(self) -> None:
        if self.start_step < 0:
            raise ValueError("event start_step must be non-negative")
        if self.end_step is not None and self.end_step <= self.start_step:
            raise ValueError("event end_step must be greater than start_step")
        if self.shape not in {"step", "ramp", "pulse", "intermittent"}:
            raise ValueError(f"unsupported schedule shape: {self.shape}")
        if self.shape == "ramp" and (self.ramp_steps is None or self.ramp_steps <= 0):
            raise ValueError("ramp schedules require ramp_steps > 0")
        if self.shape == "pulse" and self.end_step is None:
            raise ValueError("pulse schedules require end_step")
        if self.shape == "intermittent":
            if self.period is None or self.period <= 0:
                raise ValueError("intermittent schedules require period > 0")
            if not 0.0 < self.duty_cycle <= 1.0:
                raise ValueError("duty_cycle must be in (0, 1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EventSchedule:
        shape_value = str(value.get("shape", "step"))
        if shape_value not in {"step", "ramp", "pulse", "intermittent"}:
            raise ValueError(f"unsupported event schedule shape: {shape_value}")
        return cls(
            start_step=int(value.get("start_step", value.get("start", 0))),
            end_step=(
                int(value["end_step"])
                if value.get("end_step") is not None
                else int(value["end"])
                if value.get("end") is not None
                else None
            ),
            shape=cast(ScheduleShape, shape_value),
            ramp_steps=(int(value["ramp_steps"]) if value.get("ramp_steps") is not None else None),
            period=(int(value["period"]) if value.get("period") is not None else None),
            duty_cycle=float(value.get("duty_cycle", 0.5)),
        )

    def intensity(self, step: int) -> float:
        if step < self.start_step:
            return 0.0
        if self.end_step is not None and step >= self.end_step:
            return 0.0

        elapsed = step - self.start_step
        if self.shape == "ramp":
            assert self.ramp_steps is not None
            return min(1.0, (elapsed + 1) / self.ramp_steps)
        if self.shape == "intermittent":
            assert self.period is not None
            active_steps = max(1, math.ceil(self.period * self.duty_cycle))
            return 1.0 if elapsed % self.period < active_steps else 0.0
        return 1.0

    def has_finished(self, step: int) -> bool:
        return self.end_step is not None and step >= self.end_step


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    name: str
    start_step: int
    end_step: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PhaseSpec:
        return cls(
            name=str(value["name"]),
            start_step=int(value.get("start_step", value.get("start", 0))),
            end_step=(
                int(value["end_step"])
                if value.get("end_step") is not None
                else int(value["end"])
                if value.get("end") is not None
                else None
            ),
        )

    def contains(self, step: int) -> bool:
        return step >= self.start_step and (self.end_step is None or step < self.end_step)


@dataclass(slots=True)
class AgentSelector:
    """Resolve a configuration selector to stable zero-based agent indices."""

    all_agents: bool = False
    agent_indices: tuple[int, ...] = ()
    coalition_ids: tuple[str, ...] = ()
    fraction: float | None = None
    count: int | None = None
    selection_seed: int = 0
    exclude_indices: tuple[int, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> AgentSelector:
        if value is None or not value:
            return cls(all_agents=True)
        return cls(
            all_agents=bool(value.get("all", value.get("all_agents", False))),
            agent_indices=tuple(int(item) for item in value.get("agent_indices", ())),
            coalition_ids=tuple(str(item) for item in value.get("coalition_ids", ())),
            fraction=(float(value["fraction"]) if value.get("fraction") is not None else None),
            count=(int(value["count"]) if value.get("count") is not None else None),
            selection_seed=int(value.get("selection_seed", 0)),
            exclude_indices=tuple(int(item) for item in value.get("exclude_indices", ())),
        )

    def resolve(self, agents: Sequence[DroneAgent], *, event_seed: int) -> frozenset[int]:
        indices = {int(agent.agent_index) for agent in agents}
        if not self.all_agents:
            chosen: set[int] = set(self.agent_indices)
            if self.coalition_ids:
                coalition_set = set(self.coalition_ids)
                chosen.update(
                    int(agent.agent_index)
                    for agent in agents
                    if agent.coalition_id in coalition_set
                )
            if self.fraction is not None or self.count is not None:
                if self.fraction is not None and not 0.0 <= self.fraction <= 1.0:
                    raise ValueError("selector fraction must be in [0, 1]")
                available = sorted(indices)
                target_count = (
                    int(math.ceil(len(available) * self.fraction))
                    if self.fraction is not None
                    else int(self.count or 0)
                )
                target_count = min(max(target_count, 0), len(available))
                rng = np.random.default_rng(
                    stable_seed(event_seed, "selector", self.selection_seed)
                )
                if target_count:
                    selected = rng.choice(available, size=target_count, replace=False)
                    chosen.update(int(item) for item in selected)
            indices = chosen

        indices.difference_update(self.exclude_indices)
        unknown = indices.difference(int(agent.agent_index) for agent in agents)
        if unknown:
            raise ValueError(f"event selector refers to unknown agent indices: {sorted(unknown)}")
        return frozenset(indices)


def resolve_assignments(
    spec: Mapping[str, Any],
    agents: Sequence[DroneAgent],
    target_indices: frozenset[int],
) -> dict[int, str]:
    """Resolve an assignment strategy to ``agent_index -> label``."""

    if not spec:
        return {}

    current = {int(agent.agent_index): str(agent.coalition_id or "unassigned") for agent in agents}
    strategy = str(spec.get("strategy", "explicit"))
    selected = sorted(target_indices)

    if strategy == "single":
        label = str(spec.get("label", "all"))
        return {index: label for index in selected}

    if strategy in {"chunks", "index_mod"}:
        labels = [str(label) for label in spec.get("labels", ())]
        if not labels:
            groups = int(spec.get("groups", 2))
            labels = [f"group-{index}" for index in range(groups)]
        if not labels:
            raise ValueError("assignment strategy requires at least one label")
        if strategy == "index_mod":
            return {
                index: labels[position % len(labels)] for position, index in enumerate(selected)
            }

        chunks = np.array_split(np.asarray(selected, dtype=int), len(labels))
        result: dict[int, str] = {}
        for label, chunk in zip(labels, chunks, strict=True):
            result.update({int(index): label for index in chunk.tolist()})
        return result

    if strategy == "swap":
        result = {index: current[index] for index in selected}
        for pair in spec.get("swaps", ()):
            if len(pair) != 2:
                raise ValueError("each swap must contain two agent indices")
            left, right = int(pair[0]), int(pair[1])
            if left not in current or right not in current:
                raise ValueError(f"swap refers to unknown indices: {pair}")
            result[left], result[right] = current[right], current[left]
        return result

    if strategy == "relabel":
        mapping = {str(key): str(value) for key, value in spec.get("mapping", {}).items()}
        return {index: mapping.get(current[index], current[index]) for index in selected}

    # Explicit form accepts either ``groups: {label: [indices...]}`` or a direct
    # mapping of labels to index lists.
    groups_value = spec.get("groups")
    groups_mapping: Mapping[Any, Any] = groups_value if isinstance(groups_value, Mapping) else spec
    ignored = {"strategy", "labels", "groups", "swaps", "mapping"}
    result = {}
    for label, values in groups_mapping.items():
        if label in ignored:
            continue
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for index in values:
            agent_index = int(index)
            if agent_index in target_indices:
                result[agent_index] = str(label)
    return result


@dataclass(slots=True)
class ScheduledEvent:
    event_id: str
    kind: str
    schedule: EventSchedule
    selector: AgentSelector = field(default_factory=lambda: AgentSelector(all_agents=True))
    parameters: dict[str, Any] = field(default_factory=dict)
    intent: str = "fault"
    description: str = ""
    restore: bool = False
    event_seed: int = 0
    target_agent_indices: frozenset[int] = frozenset()
    _snapshot: dict[int, dict[str, Any]] = field(default_factory=dict, repr=False)
    _desired_assignments: dict[int, str] | None = field(default=None, repr=False)
    _configuration_started: bool = field(default=False, repr=False)
    _restored: bool = field(default=False, repr=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ScheduledEvent:
        event_id = str(value.get("event_id", value.get("id", ""))).strip()
        if not event_id:
            raise ValueError("every event requires a non-empty id")
        kind = str(value.get("kind", "")).strip()
        if kind not in SUPPORTED_EVENT_KINDS:
            raise ValueError(
                f"unsupported event kind {kind!r}; supported kinds: {sorted(SUPPORTED_EVENT_KINDS)}"
            )
        schedule_value = value.get("schedule", {})
        if not isinstance(schedule_value, Mapping):
            raise TypeError("event schedule must be a mapping")
        targets_value = value.get("targets")
        if targets_value is not None and not isinstance(targets_value, Mapping):
            raise TypeError("event targets must be a mapping")
        parameters = value.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise TypeError("event parameters must be a mapping")
        return cls(
            event_id=event_id,
            kind=kind,
            schedule=EventSchedule.from_mapping(schedule_value),
            selector=AgentSelector.from_mapping(targets_value),
            parameters=dict(parameters),
            intent=str(value.get("intent", "fault")),
            description=str(value.get("description", "")),
            restore=bool(value.get("restore", False)),
        )

    @property
    def stage(self) -> str:
        return _EVENT_STAGE[self.kind]

    def bind(self, model: DroneSwarmModel, *, perturbation_seed: int) -> None:
        agents = list(model.agents)
        self.event_seed = stable_seed(perturbation_seed, self.event_id)
        self.target_agent_indices = self.selector.resolve(agents, event_seed=self.event_seed)

    def intensity(self, step: int) -> float:
        return self.schedule.intensity(step)

    def affects(self, agent: DroneAgent) -> bool:
        return int(agent.agent_index) in self.target_agent_indices

    def active_for(self, step: int, agent: DroneAgent | None = None) -> bool:
        if self.intensity(step) <= 0.0:
            return False
        return agent is None or self.affects(agent)

    def _capture_snapshot(self, agents: Sequence[DroneAgent]) -> None:
        if self._snapshot:
            return
        for agent in agents:
            if not self.affects(agent):
                continue
            self._snapshot[int(agent.agent_index)] = {
                "coalition_id": agent.coalition_id,
                "role": agent.role,
                "formation_id": agent.formation_id,
                "target_position": (
                    None
                    if agent.target_position is None
                    else np.asarray(agent.target_position).copy()
                ),
                "target_id": agent.target_id,
                "restrict_interactions_to_coalition": agent.restrict_interactions_to_coalition,
            }

    def _restore_snapshot(self, agents: Sequence[DroneAgent]) -> None:
        for agent in agents:
            snapshot = self._snapshot.get(int(agent.agent_index))
            if snapshot is None:
                continue
            agent.coalition_id = snapshot["coalition_id"]
            agent.role = snapshot["role"]
            agent.formation_id = snapshot["formation_id"]
            target = snapshot["target_position"]
            agent.target_position = (
                None if target is None else np.asarray(target, dtype=float).copy()
            )
            agent.target_id = snapshot["target_id"]
            agent.restrict_interactions_to_coalition = bool(
                snapshot["restrict_interactions_to_coalition"]
            )

    def apply_configuration(self, model: DroneSwarmModel) -> None:
        if self.kind not in {"target_shift", "coalition_reconfigure"}:
            return

        agents = list(model.agents)
        intensity = self.intensity(model.tick)
        if intensity <= 0.0:
            if (
                self.restore
                and self._configuration_started
                and not self._restored
                and self.schedule.has_finished(model.tick)
            ):
                self._restore_snapshot(agents)
                self._restored = True
            return

        self._capture_snapshot(agents)
        self._configuration_started = True

        assignments_spec = self.parameters.get("assignments", {})
        if self.kind == "coalition_reconfigure" and self._desired_assignments is None:
            if not isinstance(assignments_spec, Mapping):
                raise TypeError("coalition assignments must be a mapping")
            self._desired_assignments = resolve_assignments(
                assignments_spec,
                agents,
                self.target_agent_indices,
            )

        group_targets = self.parameters.get("group_targets", {})
        if group_targets and not isinstance(group_targets, Mapping):
            raise TypeError("group_targets must be a mapping")
        individual_targets = self.parameters.get("agent_targets", {})
        if individual_targets and not isinstance(individual_targets, Mapping):
            raise TypeError("agent_targets must be a mapping")
        global_target = self.parameters.get("target_position")

        for agent in agents:
            if not self.affects(agent):
                continue
            index = int(agent.agent_index)
            if self._desired_assignments and index in self._desired_assignments:
                agent.coalition_id = self._desired_assignments[index]

            desired_target: object | None = None
            if str(index) in individual_targets:
                desired_target = individual_targets[str(index)]
            elif index in individual_targets:
                desired_target = individual_targets[index]
            elif agent.coalition_id is not None and agent.coalition_id in group_targets:
                desired_target = group_targets[agent.coalition_id]
            elif global_target is not None:
                desired_target = global_target

            if desired_target is not None:
                target = as_vector3(cast(ArrayLike, desired_target), name="event target_position")
                original = self._snapshot[index]["target_position"]
                if original is None or intensity >= 1.0:
                    agent.target_position = target
                else:
                    agent.target_position = np.asarray(original) + intensity * (
                        target - np.asarray(original)
                    )
                agent.target_id = str(
                    self.parameters.get(
                        "target_id",
                        f"{self.event_id}:{agent.coalition_id or index}",
                    )
                )

            if "restrict_interactions_to_coalition" in self.parameters:
                agent.restrict_interactions_to_coalition = bool(
                    self.parameters["restrict_interactions_to_coalition"]
                )
            if "formation_id" in self.parameters:
                agent.formation_id = str(self.parameters["formation_id"])

    def to_record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.kind,
            "intent": self.intent,
            "apply_stage": self.stage,
            "start_step": self.schedule.start_step,
            "end_step": self.schedule.end_step,
            "schedule": self.schedule.shape,
            "ramp_steps": self.schedule.ramp_steps,
            "period": self.schedule.period,
            "duty_cycle": self.schedule.duty_cycle,
            "target_agent_indices": json.dumps(sorted(self.target_agent_indices)),
            "severity_json": json.dumps(self.parameters, sort_keys=True, default=str),
            "restore": self.restore,
            "description": self.description,
        }


@dataclass(slots=True)
class EventManager:
    """Coordinate event hooks at observation, policy, actuator and physics stages."""

    events: list[ScheduledEvent] = field(default_factory=list)
    perturbation_seed: int = 0
    phases: tuple[PhaseSpec, ...] = ()
    _bound: bool = False

    @classmethod
    def from_config(
        cls,
        event_values: Sequence[Mapping[str, Any]] | None,
        *,
        perturbation_seed: int,
        phases: Sequence[Mapping[str, Any]] | None = None,
    ) -> EventManager:
        events = [ScheduledEvent.from_mapping(value) for value in event_values or ()]
        ids = [event.event_id for event in events]
        if len(ids) != len(set(ids)):
            raise ValueError("event ids must be unique within a scenario")
        phase_specs = tuple(PhaseSpec.from_mapping(value) for value in phases or ())
        return cls(events=events, perturbation_seed=perturbation_seed, phases=phase_specs)

    def bind(self, model: DroneSwarmModel) -> None:
        for event in self.events:
            event.bind(model, perturbation_seed=self.perturbation_seed)
        self._bound = True

    def before_step(self, model: DroneSwarmModel) -> None:
        if not self._bound:
            self.bind(model)
        for event in self.events:
            event.apply_configuration(model)

    def after_step(self, model: DroneSwarmModel) -> None:
        del model

    def phase_for_step(self, step: int) -> str:
        for phase in self.phases:
            if phase.contains(step):
                return phase.name
        if any(event.intensity(step) > 0.0 for event in self.events):
            return "event"
        return "unspecified"

    def active_events(
        self,
        step: int,
        *,
        agent: DroneAgent | None = None,
        stage: str | None = None,
    ) -> list[ScheduledEvent]:
        return [
            event
            for event in self.events
            if (stage is None or event.stage == stage) and event.active_for(step, agent)
        ]

    def effective_perception_radius(
        self,
        model: DroneSwarmModel,
        agent: DroneAgent,
    ) -> float:
        radius = model.perception_radius
        for event in self.active_events(model.tick, agent=agent, stage="communication"):
            if event.kind != "perception_radius_scale":
                continue
            target_scale = float(event.parameters.get("scale", 1.0))
            radius *= _interpolate(1.0, target_scale, event.intensity(model.tick))
        return max(radius, 1e-9)

    def filter_neighbors(
        self,
        model: DroneSwarmModel,
        agent: DroneAgent,
        neighbors: Sequence[DroneAgent],
        distances: Sequence[float],
    ) -> tuple[list[DroneAgent], list[float]]:
        pairs = list(zip(neighbors, distances, strict=False))
        if agent.restrict_interactions_to_coalition and agent.coalition_id is not None:
            pairs = [
                (neighbor, float(distance))
                for neighbor, distance in pairs
                if neighbor.coalition_id == agent.coalition_id
            ]

        for event in self.active_events(model.tick, agent=agent, stage="communication"):
            intensity = event.intensity(model.tick)
            if event.kind == "communication_dropout":
                probability = float(event.parameters.get("probability", 0.0)) * intensity
                kept: list[tuple[DroneAgent, float]] = []
                for neighbor, distance in pairs:
                    uniform = _stable_uniform(
                        event.event_seed,
                        "edge",
                        model.tick,
                        agent.agent_index,
                        neighbor.agent_index,
                    )
                    if uniform >= probability:
                        kept.append((neighbor, float(distance)))
                pairs = kept
            elif event.kind == "communication_partition":
                cross_coalition = bool(event.parameters.get("cross_coalition", True))
                blocked_pairs = {
                    frozenset((str(left), str(right)))
                    for left, right in event.parameters.get("blocked_pairs", ())
                }
                kept = []
                for neighbor, distance in pairs:
                    agent_group = str(agent.coalition_id)
                    neighbor_group = str(neighbor.coalition_id)
                    blocked = (cross_coalition and agent_group != neighbor_group) or frozenset(
                        (agent_group, neighbor_group)
                    ) in blocked_pairs
                    if blocked and intensity < 1.0:
                        blocked = (
                            _stable_uniform(
                                event.event_seed,
                                "partition-edge",
                                model.tick,
                                agent.agent_index,
                                neighbor.agent_index,
                            )
                            < intensity
                        )
                    if not blocked:
                        kept.append((neighbor, float(distance)))
                pairs = kept

        return [pair[0] for pair in pairs], [pair[1] for pair in pairs]

    def transform_observation(
        self,
        model: DroneSwarmModel,
        agent: DroneAgent,
        state: DroneState,
        *,
        sample_key: str,
    ) -> DroneState:
        transformed = state
        for event in self.active_events(model.tick, agent=agent, stage="observation"):
            intensity = event.intensity(model.tick)
            params = event.parameters

            if event.kind == "observation_noise":

                def noisy(
                    value: Vector3 | None,
                    field_name: str,
                    parameter_name: str,
                    *,
                    event_seed: int = event.event_seed,
                    event_params: Mapping[str, Any] = params,
                    event_intensity: float = intensity,
                ) -> Vector3 | None:
                    if value is None:
                        return None
                    std = _as_float_vector(event_params.get(parameter_name, 0.0)) * event_intensity
                    noise = _stable_normal(
                        event_seed,
                        3,
                        "observation",
                        model.tick,
                        agent.agent_index,
                        sample_key,
                        field_name,
                    )
                    return vector3(np.asarray(value, dtype=float) + std * noise)

                transformed = replace(
                    transformed,
                    position=noisy(transformed.position, "position", "position_std")
                    or transformed.position,
                    velocity=noisy(transformed.velocity, "velocity", "velocity_std")
                    or transformed.velocity,
                    local_centroid=noisy(
                        transformed.local_centroid,
                        "local_centroid",
                        "centroid_std",
                    ),
                    local_average_velocity=noisy(
                        transformed.local_average_velocity,
                        "local_average_velocity",
                        "average_velocity_std",
                    ),
                    local_separation=noisy(
                        transformed.local_separation,
                        "local_separation",
                        "separation_std",
                    ),
                    target_vector=noisy(
                        transformed.target_vector,
                        "target_vector",
                        "target_std",
                    ),
                )

            elif event.kind == "observation_bias":

                def biased(
                    value: Vector3 | None,
                    name: str,
                    *,
                    event_params: Mapping[str, Any] = params,
                    event_intensity: float = intensity,
                ) -> Vector3 | None:
                    if value is None:
                        return None
                    bias = _as_float_vector(event_params.get(name, 0.0)) * event_intensity
                    return vector3(np.asarray(value, dtype=float) + bias)

                transformed = replace(
                    transformed,
                    position=biased(transformed.position, "position_bias") or transformed.position,
                    velocity=biased(transformed.velocity, "velocity_bias") or transformed.velocity,
                    local_centroid=biased(transformed.local_centroid, "centroid_bias"),
                    local_average_velocity=biased(
                        transformed.local_average_velocity,
                        "average_velocity_bias",
                    ),
                    local_separation=biased(
                        transformed.local_separation,
                        "separation_bias",
                    ),
                    target_vector=biased(transformed.target_vector, "target_bias"),
                )

            elif event.kind == "observation_quantization":

                def quantized(
                    value: Vector3 | None,
                    name: str,
                    *,
                    event_params: Mapping[str, Any] = params,
                    event_intensity: float = intensity,
                ) -> Vector3 | None:
                    if value is None:
                        return None
                    step_value = event_params.get(name, event_params.get("step", 0.0))
                    step_array = _as_float_vector(step_value) * event_intensity
                    array = np.asarray(value, dtype=float).copy()
                    mask = step_array > 0
                    array[mask] = np.round(array[mask] / step_array[mask]) * step_array[mask]
                    return vector3(array)

                transformed = replace(
                    transformed,
                    position=quantized(transformed.position, "position_step")
                    or transformed.position,
                    velocity=quantized(transformed.velocity, "velocity_step")
                    or transformed.velocity,
                    local_centroid=quantized(transformed.local_centroid, "centroid_step"),
                    local_average_velocity=quantized(
                        transformed.local_average_velocity,
                        "average_velocity_step",
                    ),
                    local_separation=quantized(
                        transformed.local_separation,
                        "separation_step",
                    ),
                    target_vector=quantized(transformed.target_vector, "target_step"),
                )

        velocity = np.asarray(transformed.velocity, dtype=float)
        return replace(transformed, speed=float(np.linalg.norm(velocity)))

    def transform_command(
        self,
        model: DroneSwarmModel,
        agent: DroneAgent,
        action: DroneAction,
    ) -> DroneAction:
        command = action
        weight_events = [
            event
            for event in self.active_events(model.tick, agent=agent, stage="policy")
            if event.kind == "policy_weight_scale"
        ]
        if weight_events and command.components:
            component_weights = {
                "cohesion": float(getattr(agent.policy, "cohesion_weight", 0.0)),
                "alignment": float(getattr(agent.policy, "alignment_weight", 0.0)),
                "separation": float(getattr(agent.policy, "separation_weight", 0.0)),
                "goal": float(getattr(agent.policy, "goal_weight", 0.0)),
                "boundary": float(getattr(agent.policy, "boundary_weight", 0.0)),
            }
            scales = {name: 1.0 for name in component_weights}
            active_ids: list[str] = []
            for event in weight_events:
                active_ids.append(event.event_id)
                intensity = event.intensity(model.tick)
                requested = event.parameters.get("scales", event.parameters)
                if not isinstance(requested, Mapping):
                    raise TypeError("policy_weight_scale parameters must be a mapping")
                for name in scales:
                    if name in requested:
                        scales[name] *= _interpolate(1.0, float(requested[name]), intensity)

            raw = np.zeros(3, dtype=float)
            for name, weight in component_weights.items():
                component = command.components.get(name)
                if component is not None:
                    raw += weight * scales[name] * np.asarray(component, dtype=float)
            clipped = np.linalg.norm(raw) > model.max_acceleration
            acceleration = clip_norm(raw, model.max_acceleration)
            command = replace(
                command,
                acceleration=vector3(acceleration),
                raw_acceleration=vector3(raw),
                clipped=bool(clipped),
                metadata={**command.metadata, "policy_events": active_ids},
            )

        for event in self.active_events(model.tick, agent=agent, stage="policy"):
            intensity = event.intensity(model.tick)
            if event.kind == "random_action":
                scale = float(event.parameters.get("scale", model.max_acceleration))
                random_vector = (
                    _stable_normal(
                        event.event_seed,
                        3,
                        "random_action",
                        model.tick,
                        agent.agent_index,
                    )
                    * scale
                )
                random_vector = clip_norm(random_vector, model.max_acceleration)
                raw = (1.0 - intensity) * np.asarray(
                    command.acceleration
                ) + intensity * random_vector
                acceleration = clip_norm(raw, model.max_acceleration)
                command = DroneAction(
                    acceleration=vector3(acceleration),
                    action_type="random_acceleration",
                    clipped=bool(np.linalg.norm(raw) > model.max_acceleration),
                    raw_acceleration=vector3(raw),
                    components=command.components,
                    metadata={**command.metadata, "policy_event": event.event_id},
                )
            elif event.kind == "hold_command":
                raw = (1.0 - intensity) * np.asarray(command.acceleration, dtype=float)
                command = DroneAction(
                    acceleration=vector3(raw),
                    action_type="hold" if intensity >= 1.0 else command.action_type,
                    clipped=False,
                    raw_acceleration=vector3(raw),
                    components=command.components,
                    metadata={**command.metadata, "policy_event": event.event_id},
                )
        return command

    def transform_applied_action(
        self,
        model: DroneSwarmModel,
        agent: DroneAgent,
        command: DroneAction,
    ) -> DroneAction:
        actuator_events = self.active_events(model.tick, agent=agent, stage="actuator")
        if not actuator_events:
            return command

        acceleration = np.asarray(command.acceleration, dtype=float).copy()
        active_ids: list[str] = []
        saturation_limit = model.max_acceleration

        for event in actuator_events:
            active_ids.append(event.event_id)
            intensity = event.intensity(model.tick)
            params = event.parameters
            if event.kind == "actuator_gain":
                target_gain = float(params.get("gain", 1.0))
                acceleration *= _interpolate(1.0, target_gain, intensity)
            elif event.kind == "actuator_noise":
                std = _as_float_vector(params.get("std", 0.0)) * intensity
                acceleration += std * _stable_normal(
                    event.event_seed,
                    3,
                    "actuator_noise",
                    model.tick,
                    agent.agent_index,
                )
            elif event.kind == "actuator_bias":
                acceleration += _as_float_vector(params.get("bias", 0.0)) * intensity
            elif event.kind == "actuator_stuck_axis":
                fixed_value = float(params.get("value", 0.0))
                axes = [int(axis) for axis in params.get("axes", ())]
                for axis in axes:
                    if axis not in {0, 1, 2}:
                        raise ValueError("actuator_stuck_axis axes must be 0, 1, or 2")
                    acceleration[axis] = _interpolate(acceleration[axis], fixed_value, intensity)
            elif event.kind == "actuator_saturation":
                fraction = float(params.get("fraction", 1.0))
                saturation_limit = min(
                    saturation_limit,
                    model.max_acceleration * _interpolate(1.0, fraction, intensity),
                )

        raw = acceleration.copy()
        acceleration = clip_norm(acceleration, max(0.0, saturation_limit))
        return DroneAction(
            acceleration=vector3(acceleration),
            action_type=command.action_type,
            clipped=bool(np.linalg.norm(raw) > saturation_limit),
            raw_acceleration=vector3(raw),
            components=command.components,
            metadata={
                **command.metadata,
                **({"actuator_events": active_ids} if active_ids else {}),
            },
        )

    def environment_acceleration(
        self,
        model: DroneSwarmModel,
        agent: DroneAgent,
    ) -> np.ndarray:
        acceleration = np.zeros(3, dtype=float)
        for event in self.active_events(model.tick, agent=agent, stage="physics"):
            if event.kind != "wind":
                continue
            intensity = event.intensity(model.tick)
            acceleration += _as_float_vector(event.parameters.get("vector", 0.0)) * intensity
            std = _as_float_vector(event.parameters.get("std", 0.0)) * intensity
            if np.any(std > 0):
                common_mode = bool(event.parameters.get("common_mode", True))
                agent_key: object = "common" if common_mode else agent.agent_index
                acceleration += std * _stable_normal(
                    event.event_seed,
                    3,
                    "wind",
                    model.tick,
                    agent_key,
                )
        return acceleration

    def transition_context(
        self,
        model: DroneSwarmModel,
        agent: DroneAgent,
    ) -> dict[str, Any]:
        active = self.active_events(model.tick, agent=agent)
        return {
            "phase": self.phase_for_step(model.tick),
            "active_event_ids": tuple(event.event_id for event in active),
            "agent_affected": bool(active),
            "coalition_truth": agent.coalition_id,
            "role_truth": agent.role,
            "formation_truth": agent.formation_id,
            "target_id": agent.target_id,
        }

    def event_records(self) -> list[dict[str, Any]]:
        return [event.to_record() for event in self.events]
