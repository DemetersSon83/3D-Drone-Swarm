from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mesa")

from drone_swarm.mdp import transition_to_row  # noqa: E402
from drone_swarm.scenario import build_model, load_scenario, resolve_run_seeds  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("treatment_name", "control_name"),
    [
        ("actuator_noise_ramp.yaml", "nominal_boids.yaml"),
        ("communication_dropout_step.yaml", "nominal_boids.yaml"),
        ("observation_noise_ramp.yaml", "nominal_boids.yaml"),
        ("split_step.yaml", "nominal_boids.yaml"),
        ("wind_pulse.yaml", "nominal_boids.yaml"),
        ("merge_step.yaml", "merge_sham.yaml"),
        ("membership_swap.yaml", "membership_swap_sham.yaml"),
    ],
)
def test_paired_scenarios_have_identical_pre_event_dynamics(
    treatment_name: str,
    control_name: str,
) -> None:
    config_root = REPOSITORY_ROOT / "configs/experiments"
    treatment = load_scenario(config_root / treatment_name)
    control = load_scenario(config_root / control_name)
    seeds = resolve_run_seeds(123)

    treatment_model = build_model(
        treatment,
        seeds=seeds,
        run_id="treatment",
        retain_transitions=True,
    )
    control_model = build_model(
        control,
        seeds=seeds,
        run_id="control",
        retain_transitions=True,
    )
    treatment_model.run_steps(5)
    control_model.run_steps(5)

    treatment_rows = [transition_to_row(value) for value in treatment_model.transition_log]
    control_rows = [transition_to_row(value) for value in control_model.transition_log]
    for row in treatment_rows + control_rows:
        row.pop("run_id", None)
        row.pop("episode_id", None)
    assert treatment_rows == control_rows
