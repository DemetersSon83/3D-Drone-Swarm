# 3D Drone Swarm — Phase I Dataset Generator

A reproducible, Mesa-based 3D drone-swarm simulator designed to create
versioned datasets for information-theoretic and dynamical analysis. The Phase I
repository focuses on **scientifically controlled data generation** and
**pipeline-ready artifacts**. Phi Spectral and Bipredictability (`P`) are not
computed in this release; they belong in a separate analysis layer so estimator
and observer choices can evolve without rerunning the simulations.

## What Phase I provides

- Deterministic scenario execution from YAML or JSON.
- Independent random streams for initialization, policy behavior, and
  perturbations.
- Scheduled observation, communication, controller, actuator, environmental,
  target, and coalition-reconfiguration events.
- Matched treatment/control scenarios with identical pre-event trajectories
  under the same base seed.
- Two synchronized interaction tokens per drone and tick:
  - controller view: observed state, commanded action, observed outcome;
  - plant view: true state, applied action, true outcome.
- Streaming Parquet, CSV, and nested JSONL output.
- A compact `agent_signals` table for routine analytics plus a canonical wide
  `transitions` table for auditability.
- Per-run provenance, checksums, schema fingerprints, fixed side-table types,
  quality reports, and atomic `_SUCCESS` markers.
- Dataset-level manifests, catalog, schema registry, Data Package descriptor,
  dataset card, DuckDB views, checksums, and validation report.
- Loading helpers for pandas, PyArrow, Polars, and DuckDB.

## Architecture

```text
scenario YAML/JSON
        │
        ▼
scenario validation ── deterministic seeds and run ID
        │
        ▼
Mesa swarm model ── scheduled perturbation/event layer
        │
        ├── canonical transitions
        ├── compact agent signals
        ├── swarm diagnostics
        ├── event truth
        └── agent registry
        │
        ▼
validated run directory with provenance and checksums
        │
        ▼
dataset catalog, schema registry, SQL views, and manifest
```

The default staged activation keeps all records aligned to the same global
simulation tick:

```text
apply configuration events
observe all agents       -> true S_t and observed S_t
select commands          -> A_commanded,t
apply actuator events    -> A_applied,t
apply environmental force
commit all movement
observe all agents       -> true S_(t+1) and observed S_(t+1)
log both interaction views
```

## Install

Python 3.12 or newer is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,pipeline,viz]"
```

The helper script performs the same setup:

```bash
scripts/bootstrap_environment.sh
```

## Command-line interface

Installing the package exposes `drone-swarm`. The same commands can be run from
a source checkout with `PYTHONPATH=src python -m drone_swarm.cli`.

Validate a scenario without running it:

```bash
drone-swarm validate-scenario --config configs/experiments/split_step.yaml
```

Resolve its deterministic run ID:

```bash
drone-swarm run-id --config configs/experiments/split_step.yaml --seed 42
```

Run one scenario:

```bash
drone-swarm run \
  --config configs/experiments/split_step.yaml \
  --seed 42 \
  --output-root outputs/swarm_dataset_v1 \
  --resume
```

Run the small CSV/JSONL smoke scenario:

```bash
make run-smoke
```

## Launch the Phase I pilot

The supplied pilot executes every treatment and control configuration over 20
paired seeds:

```bash
JOBS=4 scripts/run_phase1_pilot.sh
```

A custom matrix can be launched directly:

```bash
scripts/run_experiments.sh \
  --config-dir configs/experiments \
  --seeds 0:19 \
  --jobs 4 \
  --output-root outputs/swarm_dataset_v1
```

The launcher validates all configurations before simulation, bounds concurrency,
writes one log per task, skips completed deterministic runs under `--resume`, and
builds a strict dataset catalog after the matrix completes.

## Output layout

```text
outputs/swarm_dataset_v1/
├── _SUCCESS
├── manifest.csv
├── manifest.parquet                 # when PyArrow is available
├── dataset_catalog.json
├── datapackage.json
├── schema_registry.json
├── dataset_summary.json
├── quality_report.json
├── DATASET_CARD.md
├── checksums.sha256
├── sql/
│   └── duckdb_views.sql
├── logs/
└── raw/
    └── <run_id>/
        ├── _SUCCESS
        ├── config.json
        ├── provenance.json
        ├── artifact_manifest.json
        ├── manifest_row.json
        ├── quality_report.json
        ├── transitions.parquet      # canonical source of truth
        ├── agent_signals.parquet    # compact analysis projection
        ├── swarm_ticks.parquet
        ├── agents.parquet
        ├── events.parquet
        └── events.json
```

Runs are written to hidden partial directories and atomically finalized only
after validation. Downstream pipelines should consume only runs and datasets
containing `_SUCCESS`.

## Analysis-facing tables

### `agent_signals`

One row per `(run_id, step, agent_id)`. It contains explicit controller and plant
fields, labels, seed metadata, event state, boids components, and time values.
This is the recommended starting point for windowed mutual-information, Phi
Spectral, `P`, forecasting, classification, and visualization pipelines.

### `transitions`

The canonical wide record, preserving the full flattened transition contract and
nested JSONL representation when requested. It includes a direct `run_id` key
plus the backward-compatible `episode_id` alias. Prefixes are:

- `s_*`, `a_*`, `sp_*`: observed state, commanded action, observed outcome.
- `true_s_*`, `applied_a_*`, `true_sp_*`: true state, applied action, true outcome.
- `environment_acceleration_*`: exogenous physical forcing.

### Side tables

- `events`: intervention type, schedule, stage, severity, and resolved targets.
- `swarm_ticks`: conventional kinematic and realized graph diagnostics.
- `agents`: stable agent index and initial/final planted labels.
- `manifest`: one row per successful run, including paths and provenance keys.

See [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) for the table contract and
[`docs/PIPELINE_INTEGRATION.md`](docs/PIPELINE_INTEGRATION.md) for loading
examples.

## Loading a generated dataset

```python
from drone_swarm.loaders import DroneSwarmDataset

swarm = DroneSwarmDataset("outputs/swarm_dataset_v1")
manifest = swarm.manifest_pandas()

signals = swarm.read_pandas(
    "agent_signals",
    scenario_ids=["split_step"],
    seeds=range(5),
)
```

PyArrow and Polars can scan Parquet lazily:

```python
arrow_dataset = swarm.arrow_dataset("agent_signals")
polars_query = swarm.polars_lazy("agent_signals")
```

DuckDB users can execute the generated SQL file from the dataset root or use:

```python
import duckdb

connection = duckdb.connect()
swarm.register_duckdb(connection)
result = connection.sql("SELECT scenario_id, count(*) FROM manifest GROUP BY 1")
```

## Validation and schemas

```bash
drone-swarm validate-run outputs/swarm_dataset_v1/raw/<run_id> --level full
drone-swarm catalog --output-root outputs/swarm_dataset_v1 --strict
drone-swarm show-schema scenario --output scenario.schema.json
```

Machine-readable JSON schemas are shipped for scenarios, per-run artifact
manifests, and dataset catalogs.

## Scope boundary

Phase I produces controlled raw data and integration artifacts. It deliberately
does **not** fit mutual-information estimators, select analysis windows, compute
Phi Spectral partitions, calculate `P/H_f/H_b`, or set anomaly thresholds. Those
steps should use healthy calibration runs, freeze observer choices, and evaluate
held-out treatment schedules in the next phase.

## Documentation

- [`docs/PHASE1.md`](docs/PHASE1.md): goals, acceptance criteria, and pilot design.
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): scenario and perturbation syntax.
- [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md): tables, keys, and semantics.
- [`docs/PIPELINE_INTEGRATION.md`](docs/PIPELINE_INTEGRATION.md): analytical loading patterns.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): identity, seeds, provenance, and restart behavior.
- [`configs/analysis/phi_p_default.yaml`](configs/analysis/phi_p_default.yaml): starting observer specification for Phase II.

## Development

```bash
make test
make lint
```

The repository is research simulation software, not a flight-control stack.
