from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from drone_swarm.scenario import (
    load_scenario,
    make_run_id,
    resolve_run_seeds,
    scenario_config_hash,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_artifact_settings_do_not_change_scientific_run_identity() -> None:
    config = load_scenario(REPOSITORY_ROOT / "configs/smoke/smoke_split.yaml")
    seeds = resolve_run_seeds(42)
    changed = deepcopy(config)
    changed["output"] = {"formats": ["parquet"], "write_agent_signals": False}
    changed["run"]["transition_batch_size"] = 999
    changed["run"]["progress_interval"] = 7
    changed["analysis_split"] = "held-out"

    assert scenario_config_hash(config) != scenario_config_hash(changed)
    assert make_run_id(config, seeds) == make_run_id(changed, seeds)


def test_trajectory_setting_changes_run_identity() -> None:
    config = load_scenario(REPOSITORY_ROOT / "configs/smoke/smoke_split.yaml")
    changed = deepcopy(config)
    changed["model"]["max_speed"] = float(changed["model"]["max_speed"]) + 0.1
    seeds = resolve_run_seeds(42)
    assert make_run_id(config, seeds) != make_run_id(changed, seeds)
