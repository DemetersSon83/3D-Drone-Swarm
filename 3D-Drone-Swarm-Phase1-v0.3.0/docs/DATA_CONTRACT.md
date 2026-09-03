# Dataset and Table Contract

## Contract versions

The repository distinguishes software version from data-contract version:

| Item | Current version | Meaning |
|---|---:|---|
| Python package | `0.3.1` | Repository implementation release. |
| Dataset contract | `1.0.0` | Top-level and per-run artifact conventions. |
| `agent_signals` contract | `1.0.0` | Compact analysis table fields and order. |

A software update may retain the same data-contract version when it does not
alter published table semantics. Breaking table changes require a new contract
version and should not silently overwrite older data.

## Dataset directory

```text
<dataset-root>/
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
├── sql/duckdb_views.sql
└── raw/<run_id>/...
```

The root `_SUCCESS` marker means the most recent catalog build did not fail. It
does not replace the quality report; pipelines should inspect both.

## Run directory

```text
raw/<run_id>/
├── _SUCCESS
├── config.json
├── provenance.json
├── artifact_manifest.json
├── manifest_row.json
├── quality_report.json
├── transitions.<format>
├── agent_signals.<format>
├── swarm_ticks.<format>
├── agents.<format>
├── events.<format>
└── events.json
```

A run is first written under `raw/.<run_id>.partial-<pid>/`. The final directory
appears only after cardinality and schema validation succeeds.

## Identifiers

| Field | Scope | Description |
|---|---|---|
| `scenario_id` | condition | Human-readable scenario name. |
| `run_id` | run | Deterministic identifier derived from the scientific configuration and all seeds. |
| `episode_id` | canonical transition | Alias of `run_id` retained by the original transition schema. |
| `base_seed` | replicate | Independent paired-run seed used as the inferential replicate. |
| `initialization_seed`, `policy_seed`, `perturbation_seed` | RNG stream | Deterministic unsigned 64-bit stream seeds; Parquet physical type is `uint64`. |
| `agent_id` | model | Mesa identity, stable within one run. |
| `agent_index` | scenario | Zero-based stable index used by selectors and cross-run comparisons. |
| `step` | transition | Zero-based transition index describing state at `t` through outcome at `t+1`. |
| `tick` | snapshot | Number of completed steps at the sampled swarm snapshot. |

### Join rules

Dataset-level joins should include the run identity:

```text
manifest.run_id = agent_signals.run_id
manifest.run_id = transitions.run_id
manifest.run_id = events.run_id
manifest.run_id = agents.run_id
manifest.run_id = swarm_ticks.run_id
```

Within a run:

```text
transitions primary key:  (step, agent_id)
agent_signals primary key: (step, agent_id)
swarm_ticks primary key:  (tick)
agents primary key:       (agent_id)
events primary key:       (event_id)
```

Across the complete dataset, prepend `run_id` to each key. The canonical
`transitions` table includes both `run_id` and the backward-compatible
`episode_id` alias.

## `agent_signals`: recommended analytical entry point

Cardinality:

```text
n_drones × steps rows per run
```

The table is a fixed projection of the canonical transition record. It keeps
fields most likely to be used by windowed mutual-information, Phi Spectral,
`P`, machine-learning, forecasting, and visualization pipelines.

### Metadata and labels

- Contract, run, scenario, split, and seed fields.
- `step`, `time`, and `next_time`.
- `agent_id` and `agent_index`.
- Phase, active events, affected flag, coalition, role, formation, and target.
- Optional reward and termination fields.

### Controller-facing token

```text
observed_state_*
commanded_action_*
observed_next_state_*
```

This is the state available to the controller, the command it emits, and the
state subsequently observed.

### Plant-facing token

```text
true_state_*
applied_action_*
environment_acceleration_*
true_next_state_*
```

This is the physical state, action reaching the plant after actuator events,
additional environmental force, and resulting physical state.

Every 3D vector is flattened into `_x`, `_y`, and `_z` columns. Neighbor and
active-event lists are compact JSON strings in flat files.

## `transitions`: canonical source of truth

The canonical table retains all fields produced by the simulator, including
raw actions, action metadata, boids decomposition, observed and true local
neighborhood summaries, truth labels, and scalar auxiliary information.

Prefix semantics:

| Prefix | Meaning |
|---|---|
| `s_*` | Observed state at transition start. |
| `a_*` | Commanded action. |
| `sp_*` | Observed state after movement. |
| `true_s_*` | True physical state at transition start. |
| `applied_a_*` | Action delivered after actuator events. |
| `environment_acceleration_*` | Exogenous physical acceleration. |
| `true_sp_*` | True physical state after movement. |

`transitions.jsonl`, when requested, retains the nested logical object. Parquet
and CSV use the stable flattened representation.

## `events`

One row per configured event. The table records:

- event ID and kind;
- intended interpretation;
- application boundary;
- start/end steps and schedule shape;
- ramp or intermittent parameters;
- resolved target agent indices;
- event severity/parameters as JSON;
- whether state is restored after the event;
- description.

Event intensity is not repeated across every agent row. The event table and
schedule should be joined or evaluated against `step` when a time-varying
severity covariate is required.

## `swarm_ticks`

Sampled at `run.metrics_stride`. Metrics include:

- mean speed and minimum pairwise distance;
- collision count and centroid;
- polarization;
- radius of gyration;
- position-covariance eigenvalues and anisotropy;
- realized interaction-graph mean degree;
- connected-component count and largest-component fraction;
- graph algebraic connectivity.

A transition at `step=t` produces a post-movement physical snapshot at
`tick=t+1`. Because snapshots are sampled, not every transition has a matching
`swarm_ticks` row.

## `agents`

One row per drone. It preserves stable agent identity and both initial and final
scenario metadata: coalition, role, formation, target identifier and coordinates,
and whether interactions are restricted to the planted coalition. Time-varying
position, velocity, mode, and battery values remain in the transition tables.

## Run metadata

### `config.json`

The exact resolved scenario, source path, complete configuration hash,
scientific simulation hash, output formats, projection choice, and all seeds.

### `provenance.json`

Repository commit and dirty state, command, runtime, timestamps, host, platform,
Python version, relevant package versions, seeds, and tabular resources.

### `artifact_manifest.json`

For each table and representation:

- relative path and media type;
- file size and SHA-256;
- row and column counts;
- ordered column descriptors;
- schema fingerprint;
- within-run and dataset-global primary keys.

### `quality_report.json`

Machine-readable results from quick, standard, or full validation. Full
validation reads key columns to test uniqueness, complete step/agent coverage,
and embedded run identity.

## Dataset metadata

### `manifest`

One row per successful run, including scientific metadata, seeds, counts,
timestamps, pairing metadata, and relative paths to preferred table files.

### `schema_registry.json`

Observed schema fingerprints and representative ordered column definitions by
table. A strict catalog fails when preferred schemas disagree across runs.

### `dataset_catalog.json`

The canonical machine-readable entry point. It identifies the manifest, table
globs, preferred formats, schema registry, Data Package descriptor, SQL file,
quality report, dataset card, and checksums.

### `datapackage.json`

A lightweight Tabular Data Package descriptor for systems that understand the
Frictionless-style resource model. Repository-specific metadata is retained in
`x-*` fields.

### `checksums.sha256`

SHA-256 coverage for primary run data, run metadata, and dataset-level catalog
artifacts.

## Null and serialization conventions

- Neighbor-derived values are null when no neighbor exists.
- Target and battery fields are null when those features are disabled.
- List and mapping fields in flat files are compact JSON strings.
- A zero environmental vector means no exogenous physical acceleration.
- No actuator event normally implies commanded and applied acceleration match.
- CSV carries stable names but not authoritative physical types; use Parquet for
  typed production analysis.
