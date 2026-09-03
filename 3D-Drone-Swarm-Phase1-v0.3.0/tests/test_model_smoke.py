from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mesa")

from drone_swarm.events import EventManager  # noqa: E402
from drone_swarm.model import DroneSwarmModel  # noqa: E402


def test_staged_model_logs_one_transition_per_agent_per_step() -> None:
    model = DroneSwarmModel(
        n_drones=5,
        bounds=((0, 10), (0, 10), (0, 10)),
        seed=123,
        perception_radius=5,
        max_speed=1,
        max_acceleration=0.25,
        collect_data=False,
    )

    model.step()

    assert len(model.transition_log) == 5
    assert model.transition_count == 5
    assert model.tick == 1
    for transition in model.transition_log:
        assert transition.true_state is not None
        assert transition.applied_action is not None
        assert transition.true_next_state is not None
        assert len(transition.state.position) == 3
        assert len(transition.action.acceleration) == 3
        assert np.all(np.asarray(transition.next_state.position) >= 0)
        assert np.all(np.asarray(transition.next_state.position) <= 10)


def test_streaming_callback_can_disable_retention() -> None:
    streamed = []
    model = DroneSwarmModel(
        n_drones=3,
        bounds=((0, 10), (0, 10), (0, 10)),
        seed=123,
        collect_data=False,
        retain_transitions=False,
        transition_callback=streamed.append,
    )
    model.run_steps(2)
    assert len(model.transition_log) == 0
    assert len(streamed) == 6
    assert model.transition_count == 6


def test_event_manager_marks_affected_transitions() -> None:
    manager = EventManager.from_config(
        [
            {
                "id": "wind",
                "kind": "wind",
                "schedule": {"start": 0},
                "targets": {"all": True},
                "parameters": {"vector": [0.1, 0.0, 0.0]},
            }
        ],
        perturbation_seed=99,
    )
    model = DroneSwarmModel(
        n_drones=2,
        bounds=((0, 10), (0, 10), (0, 10)),
        seed=123,
        collect_data=False,
        event_manager=manager,
    )
    model.step()
    assert all(transition.agent_affected for transition in model.transition_log)
    assert all(transition.active_event_ids == ("wind",) for transition in model.transition_log)
    assert all(
        transition.environment_acceleration == (0.1, 0.0, 0.0)
        for transition in model.transition_log
    )


def test_random_activation_mode_runs() -> None:
    model = DroneSwarmModel(
        n_drones=3,
        bounds=((0, 10), (0, 10), (0, 10)),
        seed=123,
        activation="random",
        collect_data=False,
    )

    model.run_steps(2)

    assert len(model.transition_log) == 6
    assert model.tick == 2
