from __future__ import annotations

import json

from drone_swarm.mdp import DroneAction, DroneState, Transition, transition_to_row, vector3


def _state(position: tuple[float, float, float]) -> DroneState:
    return DroneState(
        position=position,
        velocity=(0.1, 0.2, 0.3),
        speed=0.374,
        neighbor_count=2,
        nearest_neighbor_distance=1.5,
        local_centroid=(1.0, 2.0, 3.0),
        local_average_velocity=(0.0, 0.0, 0.0),
        neighbor_ids=(7, 8),
        local_separation=(-0.1, 0.0, 0.0),
        target_vector=(5.0, 0.0, 0.0),
    )


def test_vector3_validation() -> None:
    assert vector3([1, 2, 3]) == (1.0, 2.0, 3.0)


def test_transition_to_row_contains_controller_and_plant_vectors() -> None:
    transition = Transition(
        episode_id="episode-test",
        step=7,
        agent_id=42,
        state=_state((1.0, 2.0, 3.0)),
        action=DroneAction(
            acceleration=(0.01, 0.02, 0.03),
            raw_acceleration=(0.1, 0.2, 0.3),
            clipped=True,
            components={"cohesion": (1.0, 0.0, 0.0)},
        ),
        next_state=_state((1.1, 2.2, 3.3)),
        true_state=_state((0.9, 2.0, 3.0)),
        applied_action=DroneAction(acceleration=(0.005, 0.02, 0.03)),
        environment_acceleration=(0.1, 0.0, 0.0),
        true_next_state=_state((1.05, 2.2, 3.3)),
        reward=1.0,
        phase="fault",
        active_event_ids=("wind",),
        agent_affected=True,
        coalition_truth="A",
        info={"agent_index": 4},
    )

    row = transition_to_row(transition)

    assert row["run_id"] == "episode-test"
    assert row["episode_id"] == "episode-test"
    assert row["step"] == 7
    assert row["agent_id"] == 42
    assert row["s_position_x"] == 1.0
    assert row["s_velocity_z"] == 0.3
    assert row["s_neighbor_ids"] == "[7,8]"
    assert row["a_acceleration_y"] == 0.02
    assert row["a_raw_acceleration_z"] == 0.3
    assert row["a_component_cohesion_x"] == 1.0
    assert row["sp_position_z"] == 3.3
    assert row["true_s_position_x"] == 0.9
    assert row["applied_a_acceleration_x"] == 0.005
    assert row["environment_acceleration_x"] == 0.1
    assert row["true_sp_position_x"] == 1.05
    assert row["coalition_truth"] == "A"


def test_transition_json_serializes() -> None:
    transition = Transition(
        episode_id="episode-test",
        step=0,
        agent_id=1,
        state=_state((0.0, 0.0, 0.0)),
        action=DroneAction(acceleration=(0.0, 0.0, 0.0)),
        next_state=_state((1.0, 0.0, 0.0)),
    )
    encoded = json.dumps(transition.to_dict())
    assert "episode-test" in encoded
