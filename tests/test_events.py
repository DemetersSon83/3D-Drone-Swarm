from __future__ import annotations

from types import SimpleNamespace

import pytest

from drone_swarm.events import (
    AgentSelector,
    EventSchedule,
    ScheduledEvent,
    resolve_assignments,
)


def test_ramp_schedule_is_half_open_and_monotone() -> None:
    schedule = EventSchedule(start_step=10, end_step=20, shape="ramp", ramp_steps=4)
    assert schedule.intensity(9) == 0.0
    assert schedule.intensity(10) == pytest.approx(0.25)
    assert schedule.intensity(13) == 1.0
    assert schedule.intensity(19) == 1.0
    assert schedule.intensity(20) == 0.0


def test_intermittent_schedule() -> None:
    schedule = EventSchedule(start_step=2, shape="intermittent", period=4, duty_cycle=0.5)
    assert [schedule.intensity(step) for step in range(2, 10)] == [1, 1, 0, 0, 1, 1, 0, 0]


def test_fraction_selector_is_reproducible() -> None:
    agents = [SimpleNamespace(agent_index=index, coalition_id="all") for index in range(12)]
    selector = AgentSelector(fraction=0.25, selection_seed=17)
    first = selector.resolve(agents, event_seed=123)
    second = selector.resolve(agents, event_seed=123)
    assert first == second
    assert len(first) == 3


def test_chunk_and_swap_assignments() -> None:
    agents = [
        SimpleNamespace(agent_index=index, coalition_id="A" if index < 3 else "B")
        for index in range(6)
    ]
    selected = frozenset(range(6))
    chunked = resolve_assignments(
        {"strategy": "chunks", "labels": ["left", "right"]},
        agents,
        selected,
    )
    assert [chunked[index] for index in range(6)] == ["left"] * 3 + ["right"] * 3

    swapped = resolve_assignments(
        {"strategy": "swap", "swaps": [[1, 4]]},
        agents,
        selected,
    )
    assert swapped[1] == "B"
    assert swapped[4] == "A"


def test_event_mapping_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        ScheduledEvent.from_mapping(
            {
                "id": "bad",
                "kind": "not-a-real-event",
                "schedule": {"start": 0},
            }
        )


def test_event_mapping_rejects_unknown_schedule_shape() -> None:
    with pytest.raises(ValueError):
        ScheduledEvent.from_mapping(
            {
                "id": "bad-shape",
                "kind": "wind",
                "schedule": {"start": 0, "shape": "triangle"},
            }
        )
