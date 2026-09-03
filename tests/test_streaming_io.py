from __future__ import annotations

import csv

import pytest

from drone_swarm.io import (
    StreamingRecordWriter,
    StreamingTransitionWriter,
    write_records_parquet,
)
from drone_swarm.mdp import DroneAction, DroneState, Transition


def state(x: float) -> DroneState:
    return DroneState(
        position=(x, 0.0, 0.0),
        velocity=(0.1, 0.0, 0.0),
        speed=0.1,
        neighbor_count=1,
        nearest_neighbor_distance=1.0,
        local_centroid=(1.0, 0.0, 0.0),
        local_average_velocity=(0.1, 0.0, 0.0),
        neighbor_ids=(2,),
        local_separation=(0.0, 0.0, 0.0),
    )


def test_csv_streaming_has_dual_view_columns(tmp_path) -> None:
    transition = Transition(
        episode_id="run",
        step=0,
        agent_id=1,
        state=state(0.0),
        action=DroneAction(acceleration=(0.1, 0.0, 0.0)),
        next_state=state(0.1),
        true_state=state(0.0),
        applied_action=DroneAction(acceleration=(0.08, 0.0, 0.0)),
        environment_acceleration=(0.01, 0.0, 0.0),
        true_next_state=state(0.09),
        phase="baseline",
        coalition_truth="A",
        info={"agent_index": 0},
    )
    with StreamingTransitionWriter(tmp_path, formats=("csv",), batch_size=1) as writer:
        writer.write(transition)
    assert writer.row_count == 1

    with (tmp_path / "transitions.csv").open(newline="", encoding="utf-8") as file_obj:
        row = next(csv.DictReader(file_obj))
    assert row["s_position_x"] == "0.0"
    assert row["applied_a_acceleration_x"] == "0.08"
    assert row["true_sp_position_x"] == "0.09"
    assert row["environment_acceleration_x"] == "0.01"


def test_parquet_streaming_round_trip(tmp_path) -> None:
    parquet = pytest.importorskip("pyarrow.parquet")
    transition = Transition(
        episode_id="run",
        step=0,
        agent_id=1,
        state=state(0.0),
        action=DroneAction(acceleration=(0.1, 0.0, 0.0)),
        next_state=state(0.1),
        true_state=state(0.0),
        applied_action=DroneAction(acceleration=(0.08, 0.0, 0.0)),
        environment_acceleration=(0.01, 0.0, 0.0),
        true_next_state=state(0.09),
    )
    with StreamingTransitionWriter(tmp_path, formats=("parquet",), batch_size=1) as writer:
        writer.write(transition)

    table = parquet.read_table(tmp_path / "transitions.parquet")
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["applied_a_acceleration_x"] == 0.08
    assert row["true_sp_position_x"] == 0.09


def test_parquet_seed_columns_support_full_uint64_range(tmp_path) -> None:
    parquet = pytest.importorskip("pyarrow.parquet")
    pyarrow = pytest.importorskip("pyarrow")
    template = {
        "run_id": "template",
        "base_seed": 0,
        "initialization_seed": 0,
        "policy_seed": 0,
        "perturbation_seed": 0,
    }
    record = {
        "run_id": "large-seeds",
        "base_seed": (1 << 63) + 1,
        "initialization_seed": (1 << 64) - 1,
        "policy_seed": (1 << 63) + 2,
        "perturbation_seed": (1 << 63) + 3,
    }

    with StreamingRecordWriter(
        tmp_path,
        stem="streamed_seeds",
        template=template,
        formats=("parquet",),
        batch_size=1,
    ) as writer:
        writer.write(record)

    streamed = parquet.read_table(tmp_path / "streamed_seeds.parquet")
    assert streamed.schema.field("initialization_seed").type == pyarrow.uint64()
    assert streamed.to_pylist() == [record]

    write_records_parquet(
        [record],
        tmp_path / "materialized_seeds.parquet",
        columns=tuple(template),
        template=template,
    )
    materialized = parquet.read_table(tmp_path / "materialized_seeds.parquet")
    assert materialized.schema.field("policy_seed").type == pyarrow.uint64()
    assert materialized.to_pylist() == [record]
