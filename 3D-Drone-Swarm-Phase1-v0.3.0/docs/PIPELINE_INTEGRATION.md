# Analytical Pipeline Integration

## Dataset-global keys

Every analytical table carries a deterministic `run_id`. Use the following
dataset-global keys when joining or validating records:

- `transitions` and `agent_signals`: (`run_id`, `step`, `agent_id`)
- `swarm_ticks`: (`run_id`, `tick`)
- `agents`: (`run_id`, `agent_id`)
- `events`: (`run_id`, `event_id`)

`agent_index` is the stable scenario-assignment index used to align agents across
matched treatment and control runs. `agent_id` is the simulator identifier used
for records and neighbor references within a run.

## Recommended entry points

Use these artifacts in order:

1. `dataset_catalog.json` for discovery.
2. `quality_report.json` to determine whether the dataset is admissible.
3. `manifest.parquet` or `manifest.csv` to select runs.
4. `agent_signals.parquet` for routine time-series analysis.
5. `transitions.parquet` when canonical fields or auditability are required.
6. `events`, `agents`, and `swarm_ticks` for truth labels and external validation.

Never discover runs solely by walking directories. The manifest excludes
incomplete runs and records the preferred representation of each table.

## Python loader

```python
from drone_swarm.loaders import DroneSwarmDataset

swarm = DroneSwarmDataset("outputs/swarm_dataset_v1")
manifest = swarm.manifest_pandas()
```

Filter by condition and independent replicate:

```python
signals = swarm.read_pandas(
    "agent_signals",
    scenario_ids=["split_step", "nominal_boids"],
    seeds=range(20),
    columns=[
        "run_id",
        "scenario_id",
        "base_seed",
        "step",
        "agent_index",
        "coalition_truth",
        "observed_state_velocity_x",
        "observed_state_velocity_y",
        "observed_state_velocity_z",
    ],
)
```

## PyArrow

```python
arrow_dataset = swarm.arrow_dataset(
    "agent_signals",
    scenario_ids=["communication_dropout_step"],
)

table = arrow_dataset.to_table(
    columns=[
        "run_id",
        "step",
        "agent_index",
        "observed_state_neighbor_count",
        "true_state_neighbor_count",
    ],
    filter=None,
)
```

PyArrow is appropriate when analysts need predicate pushdown, column pruning,
and interoperability with distributed engines.

## Polars

```python
query = (
    swarm.polars_lazy("agent_signals", scenario_ids=["actuator_noise_ramp"])
    .select(
        "run_id",
        "base_seed",
        "step",
        "agent_index",
        "commanded_action_acceleration_x",
        "applied_action_acceleration_x",
    )
    .filter(pl.col("step") >= 3000)
)
frame = query.collect()
```

Import Polars in the calling module:

```python
import polars as pl
```

## DuckDB

The catalog writes `sql/duckdb_views.sql`. From the dataset root:

```bash
duckdb swarm.duckdb < sql/duckdb_views.sql
```

Example SQL:

```sql
SELECT
  scenario_id,
  base_seed,
  count(*) AS rows,
  avg(observed_state_neighbor_count) AS mean_observed_degree,
  avg(true_state_neighbor_count) AS mean_true_degree
FROM agent_signals
GROUP BY 1, 2
ORDER BY 1, 2;
```

Python registration is also supported:

```python
import duckdb
from drone_swarm.loaders import DroneSwarmDataset

connection = duckdb.connect()
swarm = DroneSwarmDataset("outputs/swarm_dataset_v1")
swarm.register_duckdb(connection, prefix="swarm_")
connection.sql("SELECT * FROM swarm_manifest LIMIT 10").show()
```

## Building windowed agent matrices

Phi Spectral and related estimators generally require a time-by-agent matrix for
one feature or a collection of feature channels. The long-form table can be
pivoted without relying on Mesa IDs across runs:

```python
run = signals[signals["run_id"] == signals["run_id"].iloc[0]]
velocity_x = run.pivot(
    index="step",
    columns="agent_index",
    values="observed_state_velocity_x",
).sort_index()
```

Use `agent_index` for a stable scenario position across paired runs. Use
`agent_id` when reconstructing within-run neighbor lists.

## Controller/plant comparisons

Observation and communication disturbances can be studied through differences
between observed and true state fields. Actuator disturbances can be studied
through commanded/applied action differences:

```python
signals["actuator_error_x"] = (
    signals["applied_action_acceleration_x"]
    - signals["commanded_action_acceleration_x"]
)
signals["neighbor_count_error"] = (
    signals["observed_state_neighbor_count"]
    - signals["true_state_neighbor_count"]
)
```

These diagnostics should remain explanatory variables or validation references;
they need not be supplied to an operational detector intended to work from the
controller boundary alone.

## Joining intervention truth

The event table is intentionally normalized. Join by `run_id`, then evaluate the
schedule against `step`. For a step event:

```python
event_active = (signals["step"] >= event_start) & (
    event_end_is_null | (signals["step"] < event_end)
)
```

Ramp and intermittent intensities follow the implementation in
`drone_swarm.events.EventSchedule` and are recorded in each scenario's resolved
configuration.

## Data-integrity gate

A minimal ingestion gate should require:

```python
import json
from pathlib import Path

root = Path("outputs/swarm_dataset_v1")
quality = json.loads((root / "quality_report.json").read_text())
if quality["status"] == "fail" or not (root / "_SUCCESS").is_file():
    raise RuntimeError("dataset did not pass the generation-quality gate")
```

For archival or transfer verification:

```bash
cd outputs/swarm_dataset_v1
sha256sum --check checksums.sha256
```

## Separation of raw and derived data

Do not write Phi Spectral, `P`, learned bins, thresholds, or windowed features
into `raw/<run_id>/`. A recommended next-phase layout is:

```text
derived/
└── <analysis-config-hash>/
    ├── observer_config.json
    ├── calibration_summary.parquet
    ├── windows.parquet
    ├── mi_edges.parquet
    ├── partitions.parquet
    ├── p_agent.parquet
    ├── p_coalition.parquet
    ├── p_swarm.parquet
    ├── evaluation.parquet
    └── provenance.json
```

The analysis hash should include feature definitions, normalization, bin edges,
window/stride, MI estimator, null model, spectral safeguards, bootstrap settings,
and alarm rule. That keeps every derived result traceable to both immutable raw
runs and a frozen observer configuration.
