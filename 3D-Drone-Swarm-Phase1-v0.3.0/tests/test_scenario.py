from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drone_swarm.scenario import (
    load_scenario,
    make_position_sampler,
    make_run_id,
    resolve_run_seeds,
    scenario_config_hash,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_scenario_loads_and_hashes_stably() -> None:
    config = load_scenario(REPOSITORY_ROOT / "configs/smoke/smoke_split.yaml")
    assert config["scenario_id"] == "smoke_split"
    assert scenario_config_hash(config) == scenario_config_hash(dict(config))


def test_run_id_changes_with_seed_but_is_reproducible() -> None:
    config = load_scenario(REPOSITORY_ROOT / "configs/smoke/smoke_split.yaml")
    first = make_run_id(config, resolve_run_seeds(7))
    second = make_run_id(config, resolve_run_seeds(7))
    third = make_run_id(config, resolve_run_seeds(8))
    assert first == second
    assert first != third


def test_rng_streams_are_distinct() -> None:
    seeds = resolve_run_seeds(42)
    assert len({seeds.initialization_seed, seeds.policy_seed, seeds.perturbation_seed}) == 3


def test_gaussian_position_sampler_clips_to_bounds() -> None:
    sampler = make_position_sampler(
        {"type": "gaussian", "center": [100, 100, 100], "std": [0, 0, 0]}
    )
    assert sampler is not None
    bounds = np.asarray([[0, 10], [0, 20], [0, 30]], dtype=float)
    value = sampler(np.random.default_rng(1), bounds)
    assert np.allclose(value, [10, 20, 30])


def test_all_experiment_scenarios_validate() -> None:
    for path in sorted((REPOSITORY_ROOT / "configs/experiments").glob("*.yaml")):
        config = load_scenario(path)
        assert config["scenario_id"]


def test_negative_seeds_are_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_run_seeds(-1)


def test_seed_values_must_fit_unsigned_64_bit_storage() -> None:
    maximum = (1 << 64) - 1
    seeds = resolve_run_seeds(0, initialization_seed=maximum)
    assert seeds.initialization_seed == maximum
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        resolve_run_seeds(0, initialization_seed=maximum + 1)
