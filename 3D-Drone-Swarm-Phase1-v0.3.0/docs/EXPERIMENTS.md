# Reproducible swarm experiment pipeline

This document describes the data-generation layer for windowed mutual-information graphs, spectral coalition detection, and classical Bipredictability-style analysis. It does not prescribe a single estimator. Raw trajectories and frozen observer configuration are kept separate so window length, binning, null models, and partition stability can be re-evaluated without rerunning the simulator.

## 1. Interaction boundaries

Every transition contains two aligned views:

```text
true S_t
   └─ sensor / communication events → observed S_t
                                      └─ policy events → commanded A_t
                                                           └─ actuator events → applied A_t
                                                                                 └─ environmental acceleration
                                                                                                      ↓
true S_{t+1}
   └─ sensor / communication events → observed S_{t+1}
```

The built-in boids policy consumes only the logged observed `DroneState`. It no longer queries hidden neighbor objects. `neighbor_ids` and `local_separation` provide the information needed for the built-in controller.

Suggested classical interaction views are:

```text
controller: (s_*, a_*, sp_*)
plant:      (true_s_*, applied_a_*, true_sp_*)
```

The difference between these views can help localize a deviation to sensing/control versus actuation/physics.

## 2. Reproducibility

A base seed deterministically generates three independent streams:

- `initialization_seed`: initial positions and velocities.
- `policy_seed`: stochastic policy decisions.
- `perturbation_seed`: event selections, noise, edge dropout, and wind.

Noise and dropped communication edges use stable keyed hashes rather than a call-order-dependent global stream. A metric probe, an extra export, or worker scheduling therefore does not change the physical perturbation.

Do not launch the same `run_id` concurrently from two processes. Resume mode deliberately removes stale partial directories and restarts an incomplete run from tick zero; it is crash recovery, not checkpoint continuation.

Use the same base seed for a treatment and its explicitly matched control. The initialization, policy, model, and pre-event agent configuration must also match; sharing a seed alone is not sufficient. The supplied pairs are regression-tested to produce identical pre-event transition rows apart from `episode_id`.

| Treatment scenarios | Matched control |
|---|---|
| `actuator_noise_ramp`, `communication_dropout_step`, `observation_noise_ramp`, `split_step`, `wind_pulse` | `nominal_boids` |
| `merge_step` | `merge_sham` |
| `membership_swap` | `membership_swap_sham` |

The scenario files record these relationships with `paired_control_scenario` or `paired_treatment_scenario`, and the fields are copied into the run manifest.

## 3. Scenario schema

A scenario is YAML or JSON.

```yaml
schema_version: 1
scenario_id: example
analysis_split: pilot
run:
  steps: 8000
  metrics_stride: 10
  transition_batch_size: 20000
model:
  n_drones: 48
  bounds: [[0, 50], [0, 50], [0, 25]]
  dt: 0.25
  perception_radius: 14
  separation_distance: 2.5
  collision_radius: 0.75
  max_speed: 3
  max_acceleration: 0.5
  target_position: [25, 25, 12.5]
  activation: staged
policy:
  type: boids
initialization:
  positions: {type: gaussian, center: [25, 25, 12.5], std: [8, 8, 4]}
  velocities: {type: random}
agents:
  initial_coalitions: {strategy: chunks, labels: [A, B]}
  restrict_interactions_to_coalition: false
phases:
  - {name: warmup, start: 0, end: 1500}
  - {name: baseline, start: 1500, end: 4000}
  - {name: event, start: 4000}
events: []
output:
  formats: [parquet]
```

### Initial position samplers

- `uniform`
- `gaussian`: `center`, scalar or 3-vector `std`
- `clusters`: weighted list of `center` and `std`
- `grid`: `shape` and fractional `margin`

### Initial velocity samplers

- `random`
- `zero`
- `aligned`: `direction`, `speed_fraction`, `direction_noise`
- `gaussian`: `mean`, `std`

### Coalition assignment strategies

- `single`: one label
- `chunks`: contiguous index chunks
- `index_mod`: interleaved groups
- `explicit`: label-to-index lists
- `swap`: exchange current labels for index pairs
- `relabel`: map current labels to new labels

Agent indices are zero-based and stable. Mesa `agent_id` values are logged separately.

## 4. Schedules and selectors

Every event has a half-open interval `[start, end)` and one schedule:

- `step`: full intensity at onset.
- `ramp`: linear rise over `ramp_steps`.
- `pulse`: step with a required end.
- `intermittent`: period and duty cycle.

Selectors support:

```yaml
targets: {all: true}
targets: {agent_indices: [0, 1, 2]}
targets: {coalition_ids: [A]}
targets: {fraction: 0.25, selection_seed: 17}
targets: {count: 8, selection_seed: 17, exclude_indices: [0]}
```

Fraction/count selections are frozen when the model is built and are recorded in `events`.

## 5. Supported event kinds

| Kind | Boundary | Important parameters |
|---|---|---|
| `observation_noise` | observation | `position_std`, `velocity_std`, `centroid_std`, `average_velocity_std`, `separation_std`, `target_std` |
| `observation_bias` | observation | matching `*_bias` vectors, including `separation_bias` |
| `observation_quantization` | observation | global `step` or field-specific `*_step` |
| `perception_radius_scale` | communication | `scale` |
| `communication_dropout` | communication | edge `probability` |
| `communication_partition` | communication | `cross_coalition`, optional `blocked_pairs` |
| `policy_weight_scale` | policy | `scales` for cohesion/alignment/separation/goal/boundary |
| `random_action` | policy | acceleration `scale` |
| `hold_command` | policy | no required parameter |
| `actuator_gain` | actuator | `gain` |
| `actuator_noise` | actuator | scalar or vector `std` |
| `actuator_bias` | actuator | `bias` |
| `actuator_stuck_axis` | actuator | `axes` using 0/1/2 and fixed `value` |
| `actuator_saturation` | actuator | fraction of nominal maximum acceleration |
| `wind` | physics | `vector`, `std`, `common_mode` |
| `target_shift` | configuration | `target_position`, `group_targets`, or `agent_targets` |
| `coalition_reconfigure` | configuration | `assignments`, targets, interaction restriction, formation label |

A configuration event may set `restore: true` when it has an end step; otherwise its last configuration persists.

## 6. Output tables

### `transitions`

One row per drone per simulator tick. Core identity/truth columns include:

```text
episode_id, step, agent_id, phase
active_event_ids, agent_affected
coalition_truth, role_truth, formation_truth, target_id
```

Each state contains position, velocity, speed, neighbor count, nearest-neighbor distance, neighbor IDs, local centroid, local average velocity, local separation steering, target vector, battery, and mode. Commanded and applied actions include acceleration, unclipped acceleration, clipping state, and boids components.

### `agent_signals`

A fixed 161-column projection of the canonical transition table. It uses
semantic names such as `observed_state_*`, `commanded_action_*`,
`true_state_*`, and `applied_action_*`, and includes run/scenario/seed metadata
on every row. This is the recommended entry point for analytical pipelines;
`transitions` remains the auditable source of truth.

### `events`

One row per configured intervention: type, intent, application stage, timing, schedule, resolved target indices, and JSON parameters.

### `swarm_ticks`

Sampled conventional diagnostics:

- mean speed, minimum distance, collision count
- centroid, polarization, radius of gyration
- covariance eigenvalues and anisotropy
- realized interaction mean degree, component count, largest-component fraction, and algebraic connectivity

### `agents`

Stable identity plus initial/final coalition, role, formation, target, and interaction restriction.

### `manifest` and catalog artifacts

The manifest contains one row per independent run, including complete and
scientific configuration hashes, independent seeds, repository commit, output
formats, event types, paths, and matched-control/treatment identifiers. The
catalog build also emits a schema registry, Tabular Data Package descriptor,
DuckDB views, checksums, a dataset card, and dataset-level quality results.
Sliding windows, agents, and pairwise edges are not independent replicates; use
the run/base-seed as the inferential unit.

## 7. Analysis preparation

`configs/analysis/phi_p_default.yaml` is an explicit starting point, not a universal optimum. Fit scaling, bin edges, null floors, window length, and alarm thresholds only on healthy calibration runs, then freeze them before validation/test runs.

For spectral MI graphs, compare several feature views rather than relying on absolute position alone:

- position relative to swarm/local centroids
- velocity relative to swarm/local mean velocity
- speed, neighbor count, nearest-neighbor distance
- commanded/applied acceleration and boids components

Always save the Fiedler vector, partition labels, eigengap, within/across MI ratio, and bootstrap stability alongside the scalar spectral score. Compare planted and inferred partitions with label-invariant measures such as adjusted Rand index or variation of information.

For Bipredictability-style analysis, avoid one enormous joint symbol over the full swarm. Estimate agent-local, planted-coalition, inferred-coalition, and low-dimensional macro-swarm views. Report `P`, forward uncertainty, backward uncertainty, and their difference relative to a healthy baseline.

## 8. Running a pilot and confirmation set

```bash
# Twenty paired pilot seeds per scenario.
scripts/run_experiments.sh --seeds 0:19 --jobs 4

# A held-out confirmation range.
scripts/run_experiments.sh --seeds 100:159 --jobs 4
```

Do not tune observer settings on the held-out range. Expand low-severity or geometry-only conditions toward 90–120 independent seeds when pilot variance indicates effects near `d = 0.3`.

## 9. Storage and memory

Transitions stream in bounded batches. With 48 drones and 8,000 ticks, one run contains 384,000 rows. Parquet with Zstandard compression is the intended production format; JSONL is useful for inspection but substantially larger.
