from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mesa")

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
    assert model.tick == 1
    for transition in model.transition_log:
        assert len(transition.state.position) == 3
        assert len(transition.state.velocity) == 3
        assert len(transition.action.acceleration) == 3
        assert len(transition.next_state.position) == 3
        assert np.all(np.asarray(transition.next_state.position) >= 0)
        assert np.all(np.asarray(transition.next_state.position) <= 10)


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
