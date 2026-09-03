# Configuration sets

- `smoke/`: small, fast configurations used for installation and CI checks.
- `experiments/`: Phase I nominal, treatment, and matched-control scenarios.
- `analysis/`: frozen starting specifications for the later Phi Spectral and
  Bipredictability analysis layer. These files do not cause the simulator to
  compute those metrics.

Every scenario uses schema version 1. The machine-readable contract can be
exported with:

```bash
drone-swarm show-schema scenario --output scenario.schema.json
```

Treatment/control relationships are declared in the files themselves and are
checked by the strict dataset catalog for each base seed.
