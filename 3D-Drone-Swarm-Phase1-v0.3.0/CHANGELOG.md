# Changelog

All notable changes to this project are documented here.

## 0.3.1 — Parquet seed-schema hotfix

### Fixed

- Store base, initialization, policy, and perturbation seeds as Arrow ``uint64``
  columns so deterministic seeds above the signed-int64 ceiling can be written
  without overflow.
- Reject explicit seeds outside the supported unsigned 64-bit range.
- Add a regression test covering both streaming and materialized Parquet writes
  at the top of the uint64 range.
- Add a dependency-enabled Parquet smoke target and CI run so CSV-only smoke
  tests cannot mask Parquet schema defects.
- Force the ordinary smoke target to execute a fresh run rather than resume an
  older artifact.
- Print an unambiguous final `[PASS]` or `[FAIL]` result from the batch launcher,
  including both run failures and strict catalog failures.

## 0.3.0 — Phase I dataset generator

### Added

- Importable, restart-safe scenario execution through `drone_swarm.runner`.
- A `drone-swarm` command-line interface for running, validating, cataloging,
  and inspecting experiments.
- A compact, versioned `agent_signals` projection intended for analytical
  pipelines, while retaining the canonical wide transition table.
- Per-run artifact manifests, schema fingerprints, provenance, quality reports,
  checksums, and atomic completion markers.
- Dataset-global primary keys in every analytical table and generated catalog.
- Fixed typed templates for side tables and manifests so nominal, treatment, and
  empty-event runs retain compatible Parquet schemas.
- Dataset-level catalog, schema registry, manifest, DuckDB views, checksums,
  quality report, and summary artifacts.
- Pandas, PyArrow, Polars, and DuckDB loading helpers.
- Independent random-number streams for initialization, policy behavior, and
  perturbations.
- Scheduled observation, communication, policy, actuator, environment, and
  coalition-reconfiguration events.
- Matched control scenarios and a 20-seed Phase I pilot launcher.

### Changed

- Deterministic run IDs are based on the scientific simulation definition and
  seeds, not on output format, batch size, progress logging, or analysis split.
- The built-in boids controller acts on the state that is actually logged.
- Transition records now retain both controller-facing and plant-facing
  `(state, action, outcome)` views.

## 0.2.0 — Phi/P experiment layer

- Introduced scheduled perturbations, planted reconfigurations, dual-view
  transition logging, streaming export, scenario configuration, and batch
  launch scripts.

## 0.1.0 — Initial simulator

- Mesa-based 3D boids model with per-agent MDP transition logging.
