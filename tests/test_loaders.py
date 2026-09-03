from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from drone_swarm.dataset import build_dataset_catalog
from drone_swarm.loaders import DroneSwarmDataset


def test_pandas_loader_uses_catalog_and_filters(
    make_fake_run: Callable[..., Path],
) -> None:
    root = make_fake_run(scenario_id="nominal", base_seed=3)
    build_dataset_catalog(root, strict=True, no_parquet=True)
    dataset = DroneSwarmDataset(root)
    manifest = dataset.manifest_pandas()
    assert manifest.loc[0, "scenario_id"] == "nominal"

    signals = dataset.read_pandas(
        "agent_signals",
        scenario_ids=["nominal"],
        seeds=[3],
        columns=["run_id", "step", "agent_id", "observed_state_position_x"],
    )
    assert len(signals) == 4
    assert signals["step"].nunique() == 2
    assert signals["agent_id"].nunique() == 2
