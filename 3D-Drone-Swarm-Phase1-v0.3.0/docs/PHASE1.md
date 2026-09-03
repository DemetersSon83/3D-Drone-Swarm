# Phase I: Controlled Dataset Generation

## Objective

Phase I establishes a reproducible experimental substrate for studying whether
Phi Spectral and classical Bipredictability (`P`) can detect and distinguish
swarm reconfiguration, degradation, and recovery. Its deliverable is not a
single simulation trace; it is a **versioned collection of independent runs**
with intervention truth, stable schemas, provenance, and quality controls.

## Scientific boundary

Phase I includes:

1. Controlled nominal and perturbed swarm generation.
2. Planned coalition changes and matched no-event controls.
3. Separate controller-facing and plant-facing interaction records.
4. Stable long-form tables for agent-wise time-series analysis.
5. Dataset integrity, provenance, and pipeline integration artifacts.

Phase I excludes:

1. Mutual-information estimator selection.
2. Phi Spectral computation and recursive spectral partitioning.
3. `P`, forward uncertainty, or backward uncertainty estimation.
4. Threshold fitting, alarm logic, or comparative performance claims.
5. Post-hoc adjustment of analysis settings using treatment outcomes.

Those tasks form the analysis and validation phase. Keeping them separate means
new estimators can be evaluated against the same immutable raw runs.

## Primary experimental unit

One independent run is defined by:

```text
scientific scenario configuration + base seed
```

The base seed deterministically derives three independent streams:

```text
initialization seed
policy seed
perturbation seed
```

This supports matched treatment/control designs. A treatment and its sham control
use the same initialization and policy streams, so their trajectories are
identical until the treatment event becomes active.

## Included scenario families

The repository supplies the following initial benchmark conditions:

| Scenario | Purpose |
|---|---|
| `nominal_boids` | Healthy homogeneous reference behavior. |
| `split_step` | Planned one-to-two coalition reconfiguration. |
| `merge_step` / `merge_sham` | Planned merge and matched no-event control. |
| `membership_swap` / `membership_swap_sham` | Agent reassignment and matched control. |
| `observation_noise_ramp` | Gradually degrading state observation. |
| `communication_dropout_step` | Abrupt partial loss of perceived edges. |
| `actuator_noise_ramp` | Increasing discrepancy between commanded and applied action. |
| `wind_pulse` | Temporary common-mode environmental forcing and recovery. |
| `common_drive_control` | Similar behavior caused by a shared target rather than interaction. |

These are first-pass benchmark families. Later experiment sets should add
formation-only changes, leader handoffs, intermittent faults, local gusts,
communication partitions, agent dropout/rejoin, and combined perturbations.

## Phase I pilot

The supplied launcher runs all experiment configurations over 20 paired seeds:

```bash
JOBS=4 scripts/run_phase1_pilot.sh
```

The pilot is intended to establish:

- runtime and storage requirements;
- natural pre-event variability;
- warm-up and settling-time distributions;
- schema and output stability;
- matched-prefix correctness;
- whether planned event severities generate observable but nontrivial responses;
- which scenarios should advance to larger held-out seed sets.

Twenty seeds are not treated as a universal confirmatory sample size. The pilot
should be used to estimate between-run variability and update the main-study
power calculation.

## Acceptance criteria

Phase I is complete when all of the following hold:

### Reproducibility

- Repeating a scenario with the same scientific configuration and seeds produces
  the same run ID and trajectory data.
- Changing output format or batch size does not change scientific run identity.
- Treatment and matched control runs share an identical pre-event prefix.
- Perturbation samples do not depend on worker scheduling or unrelated logging.

### Data contract

- Every successful run contains canonical transitions, compact agent signals,
  event truth, swarm diagnostics, an agent registry, provenance, checksums, and a
  quality report.
- Per-run keys are unique and cover the expected agents and steps.
- Dataset-level schema fingerprints are consistent within each table.
- Every treatment run declaring a paired control has a matching control run for
  the same base seed.

### Operational robustness

- Output is streamed with bounded memory.
- Failed or interrupted runs never appear as successful data.
- Completed runs can be skipped safely during restart.
- A strict catalog build fails when runs are incomplete, schemas disagree, run
  IDs duplicate, or declared controls are missing.

### Pipeline readiness

- The dataset can be loaded from pandas, PyArrow, Polars, and DuckDB without
  custom filename discovery.
- A machine-readable catalog and schema registry describe every table.
- SHA-256 checksums cover all primary data and metadata artifacts.

## Recommended Phase II handoff

Phase II should consume the Phase I dataset through `agent_signals` and the
manifest, while treating `transitions` as the auditable source of truth. Observer
configuration should be learned only from healthy calibration runs, then frozen
before held-out perturbation evaluation. Warm-up should be derived by baseline
family rather than imposed as a single global constant.
