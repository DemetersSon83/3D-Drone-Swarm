from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from drone_swarm.contracts import DATASET_CONTRACT_VERSION, build_run_artifact_manifest
from drone_swarm.io import atomic_write_json, write_records_csv
from drone_swarm.mdp import DroneAction, DroneState, Transition, transition_to_row
from drone_swarm.projections import agent_signal_row


def _state(position_x: float) -> DroneState:
    return DroneState(
        position=(position_x, 0.0, 0.0),
        velocity=(0.1, 0.0, 0.0),
        speed=0.1,
        neighbor_count=1,
        nearest_neighbor_distance=1.0,
        local_centroid=(position_x + 1.0, 0.0, 0.0),
        local_average_velocity=(0.1, 0.0, 0.0),
        neighbor_ids=(2,),
        local_separation=(-1.0, 0.0, 0.0),
        target_vector=(5.0, 0.0, 0.0),
        mode="nominal",
    )


@pytest.fixture
def make_fake_run(tmp_path: Path) -> Callable[..., Path]:
    def make(
        *,
        scenario_id: str = "nominal",
        base_seed: int = 1,
        paired_control_scenario: str | None = None,
    ) -> Path:
        run_id = f"{scenario_id}__seed-{base_seed:06d}__fixture"
        output_root = tmp_path / "dataset"
        run_dir = output_root / "raw" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        transition_rows: list[dict[str, Any]] = []
        signal_rows: list[dict[str, Any]] = []
        for step in range(2):
            for agent_index, agent_id in enumerate((1, 2)):
                state = _state(float(agent_index + step))
                next_state = _state(float(agent_index + step + 1))
                action = DroneAction(
                    acceleration=(0.01, 0.0, 0.0),
                    raw_acceleration=(0.01, 0.0, 0.0),
                    components={
                        name: (0.0, 0.0, 0.0)
                        for name in ("cohesion", "alignment", "separation", "goal", "boundary")
                    },
                )
                transition = Transition(
                    episode_id=run_id,
                    step=step,
                    agent_id=agent_id,
                    state=state,
                    action=action,
                    next_state=next_state,
                    true_state=state,
                    applied_action=action,
                    environment_acceleration=(0.0, 0.0, 0.0),
                    true_next_state=next_state,
                    phase="baseline",
                    coalition_truth="all",
                    formation_truth="cloud",
                    info={"agent_index": agent_index},
                )
                transition_rows.append(transition_to_row(transition))
                signal_rows.append(
                    agent_signal_row(
                        transition,
                        context={
                            "scenario_id": scenario_id,
                            "analysis_split": "fixture",
                            "base_seed": base_seed,
                            "initialization_seed": base_seed + 10,
                            "policy_seed": base_seed + 20,
                            "perturbation_seed": base_seed + 30,
                            "dt": 0.25,
                        },
                    )
                )

        write_records_csv(transition_rows, run_dir / "transitions.csv")
        write_records_csv(signal_rows, run_dir / "agent_signals.csv")
        write_records_csv(
            [
                {"run_id": run_id, "scenario_id": scenario_id, "base_seed": base_seed, "tick": 1},
                {"run_id": run_id, "scenario_id": scenario_id, "base_seed": base_seed, "tick": 2},
            ],
            run_dir / "swarm_ticks.csv",
        )
        write_records_csv(
            [
                {
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "base_seed": base_seed,
                    "agent_id": 1,
                },
                {
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "base_seed": base_seed,
                    "agent_id": 2,
                },
            ],
            run_dir / "agents.csv",
        )
        write_records_csv(
            [],
            run_dir / "events.csv",
            columns=("run_id", "scenario_id", "base_seed", "event_id"),
        )
        atomic_write_json(run_dir / "events.json", [])

        config = {
            "contract_version": DATASET_CONTRACT_VERSION,
            "run_id": run_id,
            "config_hash": "config-fixture",
            "simulation_hash": "simulation-fixture",
            "seeds": {
                "base_seed": base_seed,
                "initialization_seed": base_seed + 10,
                "policy_seed": base_seed + 20,
                "perturbation_seed": base_seed + 30,
            },
            "scenario": {
                "schema_version": 1,
                "scenario_id": scenario_id,
                "analysis_split": "fixture",
                "paired_control_scenario": paired_control_scenario,
                "run": {"steps": 2, "metrics_stride": 1},
                "model": {"n_drones": 2, "dt": 0.25},
            },
        }
        atomic_write_json(run_dir / "config.json", config)
        artifacts = build_run_artifact_manifest(
            run_dir,
            run_id=run_id,
            scenario_id=scenario_id,
        )
        atomic_write_json(run_dir / "artifact_manifest.json", artifacts)
        atomic_write_json(
            run_dir / "provenance.json",
            {
                "contract_version": DATASET_CONTRACT_VERSION,
                "run_id": run_id,
                "scenario_id": scenario_id,
                "resources": artifacts["resources"],
            },
        )
        atomic_write_json(
            run_dir / "manifest_row.json",
            {
                "contract_version": DATASET_CONTRACT_VERSION,
                "run_id": run_id,
                "scenario_id": scenario_id,
                "analysis_split": "fixture",
                "paired_control_scenario": paired_control_scenario,
                "config_hash": "config-fixture",
                "simulation_hash": "simulation-fixture",
                "repository_commit": "fixture",
                "base_seed": base_seed,
                "initialization_seed": base_seed + 10,
                "policy_seed": base_seed + 20,
                "perturbation_seed": base_seed + 30,
                "n_drones": 2,
                "steps": 2,
                "dt": 0.25,
                "activation": "staged",
                "event_types": "[]",
                "event_count": 0,
                "transition_rows": 4,
                "agent_signal_rows": 4,
                "metrics_rows": 2,
                "formats": '["csv"]',
                "transitions_file": "transitions.csv",
                "agent_signals_file": "agent_signals.csv",
                "swarm_ticks_file": "swarm_ticks.csv",
                "agents_file": "agents.csv",
                "events_file": "events.csv",
                "status": "success",
            },
        )
        atomic_write_json(
            run_dir / "_SUCCESS",
            {
                "contract_version": DATASET_CONTRACT_VERSION,
                "run_id": run_id,
                "transition_rows": 4,
                "agent_signal_rows": 4,
                "quality_status": "pass",
            },
        )
        return output_root

    return make
