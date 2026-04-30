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
        target_vector=(5.0, 0.0, 0.0),
    )


def test_vector3_validation() -> None:
    assert vector3([1, 2, 3]) == (1.0, 2.0, 3.0)


def test_transition_to_row_contains_state_action_next_state_vectors() -> None:
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
        reward=1.0,
    )

    row = transition_to_row(transition)

    assert row["episode_id"] == "episode-test"
    assert row["step"] == 7
    assert row["agent_id"] == 42
    assert row["s_position_x"] == 1.0
    assert row["s_velocity_z"] == 0.3
    assert row["a_acceleration_y"] == 0.02
    assert row["a_raw_acceleration_z"] == 0.3
    assert row["a_component_cohesion_x"] == 1.0
    assert row["sp_position_z"] == 3.3


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
