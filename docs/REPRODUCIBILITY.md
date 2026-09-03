# Reproducibility and Run Identity

## Scientific identity versus artifact settings

The repository records two hashes:

- `simulation_hash`: fields capable of changing trajectories or labels;
- `config_hash`: the complete resolved scenario, including output settings.

The deterministic run ID is built from the simulation hash and all resolved
seeds. Output format, batch size, progress logging, analysis split, and pairing
metadata do not change that ID. This prevents a scientifically identical run
from being mistaken for a new replicate merely because it was exported
as CSV instead of Parquet.

Changing any trajectory-relevant setting—model parameters, initialization,
policy, event schedule, planted labels, or number of steps—changes the run ID.

## Random-number streams

A non-negative base seed derives independent streams using stable hashing:

```text
initialization_seed = f(base_seed, "initialization")
policy_seed         = f(base_seed, "policy")
perturbation_seed   = f(base_seed, "perturbation")
```

The streams can be overridden explicitly and are always recorded. Event noise
and communication-edge dropout use keys containing the perturbation seed, event,
step, agent, and sample role. Their realized values therefore do not depend on
parallel-worker order or unrelated random calls.

## Matched treatment/control runs

A paired treatment and control should:

1. Use the same base seed.
2. Share model, policy, initialization, and pre-event configuration.
3. Differ only in the treatment event at or after the declared onset.
4. Declare pairing metadata in the scenario files.

The test suite verifies identical pre-event records for supplied pairs. The
strict dataset catalog also reports missing same-seed controls.

## Atomic output lifecycle

```text
raw/.<run_id>.partial-<pid>/
        │
        ├── streaming tables
        ├── metadata and validation
        ├── failure.json on exception
        │
        └── atomic rename after success
                ▼
raw/<run_id>/
```

`_SUCCESS` is written last inside the partial directory. The entire directory is
then atomically renamed. Interrupted runs are never mistaken for completed runs.
`--resume` skips successful runs and restarts partial/incomplete runs from the
beginning; it does not attempt an unsafe mid-trajectory reconstruction.

## Provenance

Every run records:

- source scenario and resolved configuration;
- both configuration hashes;
- all seeds;
- repository commit and dirty state when available;
- command line;
- start/end times and runtime;
- host, platform, Python, and relevant package versions;
- output table descriptors, checksums, schemas, and cardinalities.

A repository archive without `.git` will report an unknown commit. For
publication runs, execute from a clean Git checkout or container image and keep
the resulting provenance with the dataset.

## Reproducibility checks

Before a pilot:

```bash
make test
make lint
drone-swarm validate-scenario --config configs/experiments/split_step.yaml
scripts/run_experiments.sh --config configs/smoke/smoke_split.yaml --seeds 0:1 --jobs 2 --dry-run
```

After generation:

```bash
drone-swarm catalog --output-root outputs/phase1_pilot --strict --validation-level standard
sha256sum --check outputs/phase1_pilot/checksums.sha256
```

For selected archival runs, use full key validation:

```bash
drone-swarm validate-run outputs/phase1_pilot/raw/<run_id> --level full
```
