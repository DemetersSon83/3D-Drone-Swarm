from __future__ import annotations

from types import SimpleNamespace

from drone_swarm.mdp import DroneState
from drone_swarm.policies import BoidsPolicy


def test_boids_policy_uses_logged_state_without_neighbor_query() -> None:
    state = DroneState(
        position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        speed=0.0,
        neighbor_count=2,
        nearest_neighbor_distance=0.5,
        local_centroid=(1.0, 0.0, 0.0),
        local_average_velocity=(0.0, 1.0, 0.0),
        local_separation=(-1.0, 0.0, 0.0),
        target_vector=(0.0, 0.0, 1.0),
    )
    model = SimpleNamespace(
        max_speed=2.0,
        max_acceleration=0.5,
        bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)),
        boundary_margin=0.0,
        separation_distance=1.0,
    )
    action = BoidsPolicy().select_action(state=state, agent=object(), model=model)
    assert action.components is not None
    assert set(action.components) == {"cohesion", "alignment", "separation", "goal", "boundary"}
    assert action.action_type == "avoid_neighbor"
