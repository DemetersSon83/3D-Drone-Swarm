from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from drone_swarm.dataset import build_dataset_catalog
from drone_swarm.mdp import DroneAction, DroneState, Transition
from drone_swarm.runner import RunOptions, execute_run


def _state(agent_index: int, step: int) -> DroneState:
    value = float(agent_index + step)
    return DroneState(
        position=(value, 1.0, 2.0),
        velocity=(0.1, 0.0, 0.0),
        speed=0.1,
        neighbor_count=1,
        nearest_neighbor_distance=1.0,
        local_centroid=(value + 1.0, 1.0, 2.0),
        local_average_velocity=(0.1, 0.0, 0.0),
        neighbor_ids=((agent_index + 1) % 2 + 1,),
        local_separation=(0.0, 0.0, 0.0),
        target_vector=(1.0, 0.0, 0.0),
        mode="nominal",
    )


class _FakeEventManager:
    def phase_for_step(self, step: int) -> str:
        return "baseline" if step < 2 else "event"

    def event_records(self) -> list[dict[str, Any]]:
        return [
            {
                "event_id": "fake-event",
                "event_type": "wind",
                "intent": "fault",
                "apply_stage": "physics",
                "start_step": 2,
                "end_step": None,
                "schedule": "step",
                "ramp_steps": 0,
                "period": 1,
                "duty_cycle": 1.0,
                "target_agent_indices": "[0,1]",
                "severity_json": '{"vector":[0.1,0.0,0.0]}',
                "restore": False,
                "description": "runner test",
            }
        ]


class _FakeModel:
    def __init__(
        self,
        *,
        run_id: str,
        transition_callback: Callable[[Transition], None],
    ) -> None:
        self.run_id = run_id
        self.transition_callback = transition_callback
        self.tick = 0
        self.transition_count = 0
        self.event_manager = _FakeEventManager()

    def step(self) -> None:
        for agent_index in range(2):
            state = _state(agent_index, self.tick)
            next_state = _state(agent_index, self.tick + 1)
            action = DroneAction(
                acceleration=(0.01, 0.0, 0.0),
                raw_acceleration=(0.01, 0.0, 0.0),
                components={
                    "cohesion": (0.0, 0.0, 0.0),
                    "alignment": (0.0, 0.0, 0.0),
                    "separation": (0.0, 0.0, 0.0),
                    "goal": (0.01, 0.0, 0.0),
                    "boundary": (0.0, 0.0, 0.0),
                },
            )
            self.transition_callback(
                Transition(
                    episode_id=self.run_id,
                    step=self.tick,
                    agent_id=agent_index + 1,
                    state=state,
                    action=action,
                    next_state=next_state,
                    true_state=state,
                    applied_action=action,
                    environment_acceleration=(0.0, 0.0, 0.0),
                    true_next_state=next_state,
                    phase=self.event_manager.phase_for_step(self.tick),
                    active_event_ids=("fake-event",) if self.tick >= 2 else (),
                    agent_affected=self.tick >= 2,
                    coalition_truth="all",
                    role_truth="member",
                    formation_truth="cloud",
                    target_id="target",
                    info={"agent_index": agent_index},
                )
            )
            self.transition_count += 1
        self.tick += 1

    def metrics_snapshot(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "n_drones": 2,
            "transition_count": self.transition_count,
            "mean_speed": 0.1,
            "min_pairwise_distance": 1.0,
            "collision_count": 0,
            "centroid_x": 0.5,
            "centroid_y": 1.0,
            "centroid_z": 2.0,
            "polarization": 1.0,
            "radius_of_gyration": 0.5,
            "position_eigenvalue_1": 0.25,
            "position_eigenvalue_2": 0.0,
            "position_eigenvalue_3": 0.0,
            "position_anisotropy": 1.0,
            "interaction_mean_degree": 1.0,
            "interaction_component_count": 1,
            "interaction_largest_component_fraction": 1.0,
            "interaction_algebraic_connectivity": 2.0,
        }

    def agent_registry(self) -> list[dict[str, Any]]:
        return [
            {
                "agent_id": index + 1,
                "agent_index": index,
                "coalition_id": "all",
                "role": "member",
                "formation_id": "cloud",
                "target_id": "target",
                "target_x": 1.0,
                "target_y": 2.0,
                "target_z": 3.0,
                "restrict_interactions_to_coalition": False,
            }
            for index in range(2)
        ]


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scenario_id": "runner_test",
                "analysis_split": "test",
                "run": {
                    "steps": 4,
                    "metrics_stride": 2,
                    "transition_batch_size": 3,
                },
                "model": {
                    "n_drones": 2,
                    "dt": 0.25,
                    "perception_radius": 2.0,
                },
                "policy": {"type": "boids"},
                "phases": [
                    {"name": "baseline", "start": 0, "end": 2},
                    {"name": "event", "start": 2},
                ],
                "events": [],
                "output": {"formats": ["csv"], "write_agent_signals": True},
            }
        ),
        encoding="utf-8",
    )


def test_execute_run_writes_atomic_pipeline_ready_artifacts(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    config_path = tmp_path / "scenario.json"
    _write_config(config_path)
    output_root = tmp_path / "dataset"

    def fake_build_model(
        config: Any,
        *,
        seeds: Any,
        run_id: str,
        transition_callback: Callable[[Transition], None],
        retain_transitions: bool,
    ) -> _FakeModel:
        del config, seeds, retain_transitions
        return _FakeModel(run_id=run_id, transition_callback=transition_callback)

    monkeypatch.setattr("drone_swarm.runner.build_model", fake_build_model)
    result = execute_run(
        RunOptions(
            config_path=config_path,
            output_root=output_root,
            base_seed=7,
            formats=("csv",),
            validation_level="full",
            repository_root=tmp_path,
        )
    )

    assert result["status"] == "success"
    run_directory = Path(result["path"])
    assert run_directory.is_dir()
    assert (run_directory / "_SUCCESS").is_file()
    assert not list((output_root / "raw").glob(".*.partial-*"))

    quality = json.loads((run_directory / "quality_report.json").read_text())
    assert quality["status"] == "pass"
    assert quality["path"] == str(run_directory)
    assert quality["validation_stage"] == "pre_atomic_finalize"

    with (run_directory / "transitions.csv").open(newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert len(rows) == 8
    assert {row["run_id"] for row in rows} == {result["run_id"]}

    with (run_directory / "agent_signals.csv").open(newline="", encoding="utf-8") as file_obj:
        signal_rows = list(csv.DictReader(file_obj))
    assert len(signal_rows) == 8
    assert signal_rows[0]["observed_state_position_x"] == "0.0"

    summary = build_dataset_catalog(
        output_root,
        strict=True,
        no_parquet=True,
        validation_level="full",
    )
    assert summary["quality_status"] == "pass"
    assert summary["transition_rows"] == 8
    assert (output_root / "dataset_catalog.json").is_file()

    skipped = execute_run(
        RunOptions(
            config_path=config_path,
            output_root=output_root,
            base_seed=7,
            formats=("csv",),
            resume=True,
            repository_root=tmp_path,
        )
    )
    assert skipped["status"] == "skipped"
    assert skipped["run_id"] == result["run_id"]
