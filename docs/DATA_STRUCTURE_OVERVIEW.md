# Phase I Dataset Structure Overview

## At a glance

The independent experimental unit is one **scenario × base-seed run**. Each run
contains aligned per-agent transitions, a compact analysis projection, event
truth, swarm diagnostics, an agent registry, and reproducibility metadata.
Successful runs are indexed by a dataset-level manifest and machine-readable
catalog.

```text
scenario configuration × base seed
              │
              ├── initialization seed
              ├── policy seed
              └── perturbation seed
              │
              ▼
       one independent run
              │
              ├── canonical transitions
              ├── compact agent signals
              ├── event ground truth
              ├── swarm-level telemetry
              ├── agent registry
              └── provenance and quality metadata
```

For a run with `N` drones and `T` steps:

```text
transitions rows   = N × T
agent_signals rows = N × T
agents rows        = N
swarm_ticks rows   = ceil(T / metrics_stride)
events rows        = number of configured events
```

## Top-level dataset

```text
outputs/<dataset-id>/
├── _SUCCESS
├── manifest.csv
├── manifest.parquet
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
    ├── <run-id>/
    ├── <run-id>/
    └── ...
```

`dataset_catalog.json` is the preferred machine entry point. The manifest is the
preferred run-selection table. The root `_SUCCESS` marker is written only when
the catalog quality status is not `fail`.

## Per-run directory

```text
raw/<run-id>/
├── _SUCCESS
├── config.json
├── provenance.json
├── artifact_manifest.json
├── manifest_row.json
├── quality_report.json
├── transitions.parquet
├── agent_signals.parquet
├── swarm_ticks.parquet
├── agents.parquet
├── events.parquet
└── events.json
```

CSV and nested JSONL representations can be selected in scenario configuration
or at the command line. Side tables fall back to CSV when Parquet is not
requested.

A run is built first in:

```text
raw/.<run-id>.partial-<process-id>/
```

It is renamed to the final directory only after artifact validation and the
run-level `_SUCCESS` marker are complete.

## Run identity

The run ID has the form:

```text
<scenario-id>__seed-<six-digit-base-seed>__<hash>
```

The hash includes trajectory- and label-relevant scenario content plus all
resolved random seeds. It excludes artifact-only choices such as CSV versus
Parquet, transition batch size, progress output, and analysis split.

The manifest records:

- `run_id` and `scenario_id`;
- analysis split and paired-control metadata;
- complete and scientific configuration hashes;
- base, initialization, policy, and perturbation seeds;
- drone count, steps, time increment, and activation mode;
- event types and counts;
- table row counts and preferred relative paths;
- repository commit, timestamps, runtime, and status.

## Interaction-boundary records

Each drone transition contains two synchronized state–action–outcome views:

```text
TRUE / PLANT VIEW
true state S_t
   │
   ├── communication and observation events
   ▼
CONTROLLER VIEW
observed state S_t
   │
   ├── controller and policy events
   ▼
commanded action A_t
   │
   ├── actuator events
   ▼
applied action A_t
   │
   ├── environmental acceleration and dynamics
   ▼
true state S_(t+1)
   │
   └── communication and observation events
   ▼
observed state S_(t+1)
```

This separation permits controller-facing and plant-facing `P` calculations and
helps attribute changes to sensing, control, actuation, or environment.

## `agent_signals`: 161-column compact table

`agent_signals` is the recommended starting point for most analytical pipelines.
It has one row per `(run_id, step, agent_id)` and uses explicit semantic prefixes.

### Identity and truth

```text
contract_version
run_id, scenario_id, analysis_split
base_seed, initialization_seed, policy_seed, perturbation_seed
step, time, next_time
agent_id, agent_index
phase, active_event_ids, agent_affected
coalition_truth, role_truth, formation_truth, target_id
reward, done
```

### Controller-facing token

```text
observed_state_*
commanded_action_*
observed_next_state_*
```

### Plant-facing token

```text
true_state_*
applied_action_*
environment_acceleration_*
true_next_state_*
```

Each state block contains:

```text
speed
neighbor_count
nearest_neighbor_distance
neighbor_ids
battery
mode
position_x/y/z
velocity_x/y/z
local_centroid_x/y/z
local_average_velocity_x/y/z
local_separation_x/y/z
target_vector_x/y/z
```

Each action block contains:

```text
type
clipped
acceleration_x/y/z
component_cohesion_x/y/z
component_alignment_x/y/z
component_separation_x/y/z
component_goal_x/y/z
component_boundary_x/y/z
```

## `transitions`: 163-column canonical table

The canonical flattened table preserves all simulator fields and backwards-
compatible column names. It is the audit source when a projection omits a field.
The principal prefixes are:

| Prefix | Meaning |
|---|---|
| `s_*` | Observed controller state at `t`. |
| `a_*` | Commanded action. |
| `sp_*` | Observed controller state at `t+1`. |
| `true_s_*` | True physical state at `t`. |
| `applied_a_*` | Action after actuator events. |
| `environment_acceleration_*` | Exogenous physical forcing. |
| `true_sp_*` | True physical state at `t+1`. |

When JSONL is requested, the same record is stored as a nested object rather
than 163 flattened scalar columns.

## Side tables

### `events`

One row per configured intervention, including event type, application stage,
step/ramp/pulse/intermittent schedule, resolved agent targets, severity mapping,
and restoration behavior.

### `swarm_ticks`

One row per sampled completed tick. Fields include conventional kinematic and
realized interaction-graph diagnostics such as polarization, radius of gyration,
shape anisotropy, degree, connected components, and algebraic connectivity.

### `agents`

One row per drone, preserving its stable agent index and initial/final coalition,
role, formation, target, mode, position, and velocity metadata.

## Machine-readable integration artifacts

| Artifact | Role |
|---|---|
| `dataset_catalog.json` | Discovers manifests, tables, schemas, SQL, quality, and checksums. |
| `datapackage.json` | Tabular Data Package-style interoperability descriptor. |
| `schema_registry.json` | Ordered columns, physical types, fingerprints, and consistency status. |
| `artifact_manifest.json` | Per-run file inventory, rows, columns, sizes, and SHA-256 values. |
| `quality_report.json` | Run- or dataset-level validation findings. |
| `checksums.sha256` | Transfer and archive integrity verification. |
| `sql/duckdb_views.sql` | Ready-to-execute views over all cataloged run tables. |
| `DATASET_CARD.md` | Human-readable summary generated with the catalog. |

## Recommended raw/derived separation

Phase I writes immutable generated data under `raw/`. Phi Spectral, mutual-
information edges, spectral partitions, `P`, uncertainty components, thresholds,
and performance results should be written to a separate analysis namespace:

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

This keeps every result traceable to both immutable run checksums and a frozen
analysis configuration.
