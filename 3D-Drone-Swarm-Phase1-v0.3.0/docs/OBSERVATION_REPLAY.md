# Causal observation replay

The research replay adapter converts a completed staged simulator run into a
strict pre-state observation stream and admits complete, fixed-cohort frames.
It implements observation contract `0.1.0` against simulator data contract
`1.0.0`. It does not compute persistent homology, Phi Spectral, Hodge features,
formation classifications, or forecasts, and includes no SIMPLEX assets.

Replay uses an **unpaced logical clock**. Processing duration is measured, but
this is not a live sensor driver, real-time qualification, or FPGA benchmark.

## Quick start

Run commands from this package directory after installing it. CSV replay needs
the base package; Parquet input additionally needs `pip install -e '.[parquet]'`.
A source checkout can substitute `PYTHONPATH=src python -m drone_swarm.cli` for
`drone-swarm`.

For a small reproducible source run, create this example once:

```bash
drone-swarm run \
  --config configs/smoke/smoke_split.yaml \
  --seed 42 \
  --run-id replay-example \
  --output-root outputs/replay_example_source \
  --formats csv \
  --agent-signals
```

Replay that existing run, then independently validate the emitted observations:

```bash
drone-swarm replay outputs/replay_example_source/raw/replay-example \
  --output-directory outputs/replay_example_completed

drone-swarm validate-observations \
  outputs/replay_example_completed/observations.jsonl \
  --metadata outputs/replay_example_completed/metadata.json
```

For another existing run, replace only the run-directory argument and choose a
new output directory. Output directories must not already exist and must be
outside the source run. There is no overwrite flag; source artifacts are not
modified.

The smoke scenario declares `dt = 0.25`. Replay preserves that interval; it does
not silently turn this run into a 20 Hz workload. Native `dt = 0.05` data must be
generated with an appropriate scenario for that separate workload.

To evaluate the hypothetical pre-action observation condition separately:

```bash
drone-swarm replay outputs/replay_example_source/raw/replay-example \
  --output-directory outputs/replay_example_idealized \
  --availability-basis idealized_pre_state_replay
```

Both commands print JSON reports. Exit code `0` means pass; `1` means failure.
Identical duplicates are counted and ignored. Missing/skipped frames, rejected
observations, late records, identity/frame resets, or malformed records fail the
controlled replay/validation report.

## CLI options

`drone-swarm replay RUN_DIRECTORY --output-directory OUTPUT_DIRECTORY` accepts:

| Option | Default | Meaning |
|---|---|---|
| `--availability-basis` | `completed_transition_lower_bound` | This mode or `idealized_pre_state_replay`; see timing below. |
| `--frame-id` | `simulation_cartesian` | Coordinate-frame identifier. Changing the identifier does not transform coordinates. |
| `--frame-origin` | `unspecified_legacy_simulation` | Explicit provenance describing the origin, or the default unknown declaration. |
| `--frame-handedness` | `unspecified_legacy_simulation` | Explicit provenance describing handedness, or the default unknown declaration. |
| `--max-tracks` | `128` | Maximum admitted declared cohort size. |
| `--max-queued-frames` | `4` | Maximum pending frames and recent finalized-frame duplicate-cache length. |
| `--allowed-lateness` | Source `dt` | Nonnegative allowance in simulation time, starting at nominal availability. |
| `--batch-size` | `1024` | Parquet read batch size, from 1 through 65,536; CSV is iterated row by row. |

`drone-swarm validate-observations INPUT_FILE --metadata METADATA_FILE` validates
canonical JSONL envelopes and replays causal admission in file order. It prints
its report and does not rewrite files. The presented metadata defines the
expected cohort, step count, timing, source, and coordinate declarations. This
command checks stream validity; it does not authenticate the original source
or verify the replay report's output hashes.

## Source gates and information boundary

Before replay, the adapter requires:

- A completed source run with an acceptable `_SUCCESS` quality status and
  standard run validation with checksum verification. A present failed quality
  report is rejected. Source warnings remain visible in replay metadata.
- Artifact paths confined to the run directory, valid declared column schemas,
  and CSV/Parquet headers matching the corresponding declaration.
- Explicit `staged` activation in the manifest, consistent with configuration
  when present. Sequential activation is not accepted as a synchronized frame.
- Consistent positive source `dt`, declared population, step count, and run
  identity; an `agents` registry with exactly that unique fixed cohort.
- Supported `agent_signals` and `agents` tables. Parquet is preferred when
  available; otherwise CSV is selected.
- Nondecreasing source steps within the run bounds. The adapter does not sort or
  materialize an entire input table. The consumed signal-table row count and
  final SHA-256 are checked against its manifest.

Only the following `agent_signals` columns reach the observation adapter:

| Purpose | Source fields |
|---|---|
| Contract and identity | `contract_version`, `run_id`, `step`, `agent_id` |
| Source timing | `time`, `next_time` |
| Observed pre-state position | `observed_state_position_x`, `_y`, `_z` |
| Observed pre-state velocity | `observed_state_velocity_x`, `_y`, `_z` |

The registry reader selects only `run_id` and `agent_id`. Truth states,
next-state observations, actions, rewards, interventions, coalition/role/target
labels, and scenario descriptions are excluded from canonical observations.
Source artifacts still undergo integrity checks; their hashes and run identity
are provenance, not numerical detector features. Changing excluded labels
without updating their source checksums correctly fails integrity validation.

Canonical records carry identity and timing, position/velocity and validity,
explicit quality flags, frame/clock/unit declarations, and nullable covariance.
Sample IDs are a deterministic JSON tuple of stream, step, track, and
`pre_state`; simulator track epochs start at zero. The standalone validator
checks that sample IDs follow this policy without retaining an unbounded set of
all previously seen IDs.

Unknown covariance stays `null`; it is not zero uncertainty. Vectors cannot
contain nonfinite components. Supplied covariance must be a finite, symmetric,
positive-semidefinite 3-by-3 matrix under the documented numerical tolerance in
`observations.py`. Canonical parsing rejects unknown fields, malformed values,
and duplicate JSON keys. Validity flags remain explicit: missing vectors are
null and invalid, and cannot silently become zero-valued measurements.

## Timing and causal availability

For step `k` and source interval `dt`, the source times must agree with
`time = k * dt` and `next_time = (k + 1) * dt`. After a tightly bounded source
roundoff check, canonical times use this exact declared grid. Metadata records
that normalization policy.

| Replay basis | Event time | Availability and default receipt time | Interpretation |
|---|---|---|---|
| `completed_transition_lower_bound` | `k * dt` | `(k + 1) * dt` | The completed logged row exists only after the transition. This is an assumed lower bound, not measured delivery latency. |
| `idealized_pre_state_replay` | `k * dt` | `k * dt` | Hypothetical pre-action observation callback; the current logged-data path does not supply that callback. |

The two modes remain explicitly labeled. A next-state draw is never spliced into
the following pre-state sample. Receipt/advance times must be finite,
nonnegative, and nondecreasing, and an observation cannot enter before its
availability. The assembler also rejects observations dated before the selected
mode's nominal availability.

Time and length remain `simulation_time` and `simulation_length`. Metadata
records `x, y, z` axis order, boundary convention, and explicit origin and
handedness declarations. Physical calibration and clock uncertainty remain
unknown. There is no inferred conversion to seconds, metres, ENU, or NED.
`input_samples_per_simulation_time_unit` is a logical sampling rate, not proof
of a physical sensor rate.

## Frames, gaps, and resource limits

The initial frame requires valid position **and** velocity for every declared
track at exactly one step. It is not a position-only or variable-membership
feature mode. Missing tracks do not imply that physical agents departed.
Interpolation, extrapolation, and zero filling are not performed by replay.
Explicitly supplied imputed/extrapolated records retain their quality flags and
produce a degraded quality status if otherwise valid.

The frame deadline is:

```text
nominal_available_time + allowed_lateness
```

Arrivals exactly at that deadline remain admissible while the frame is still
open; expiration happens strictly afterward. A complete frame emits immediately
without an additional hold. If a later frame completes first, unresolved earlier
steps are explicitly skipped with reason `superseded_by_complete_frame` so
outputs remain ordered. Those finalized steps cannot subsequently be revised,
even by an arrival at their original deadline.

Pending storage is limited by `max_queued_frames`; the recent duplicate cache
has the same frame bound. Under capacity pressure, skipped work is reported
explicitly. The newest complete frame is retained when available; otherwise the
oldest pending work is skipped to make room. Whole absent step ranges are
reported compactly. No emitted event history or feature-history window is kept
inside the assembler. These are record-count bounds, not a qualification of a
process RSS or FPGA memory target.

Identical recently cached observations are ignored with counters; conflicting
sample IDs or same-track/same-step records are rejected. Arrivals for finalized
frames cannot rewrite outputs. Once the bounded duplicate cache has evicted a
frame, subsequent old arrivals are classified as late rather than being compared
with indefinitely retained originals.

A coordinate/unit/clock change, source identity change for a track, or increased
track epoch triggers history invalidation and closes admission for that stream
segment. Pending frames are invalidated and the caller must construct a new
assembler. Stale lower epochs are rejected. Gaps increment the history
generation and appear on the next valid frame; downstream temporal analytics
must account for them. This implementation does not repair identity or infer
reacquisition.

Frame events keep observation quality, assembly completeness, and timeliness
separate. `computation_status = complete` means frame assembly is complete; it
does not mean that a topological or spectral feature was calculated. Frame-ready
events include event time, nominal availability, latest input availability,
receipt times, emission time, input delivery delay, queue delay, sample age,
ordered canonical observations, and gap/history-generation metadata.

## Output files

| File | Contents |
|---|---|
| `metadata.json` | Contract/producer versions, source and schema hashes, cohort, step count, units/frame/clock, replay mode, source validation status, and resource limits. |
| `observations.jsonl` | One envelope per successfully constructed canonical observation: exactly `step`, `received_time`, and `observation`. A constructed record can still fail frame admission. |
| `frames.jsonl` | Incremental frame-ready, skip, rejection, duplicate, capacity, and history-reset events. |
| `replay_report.json` | Pass/fail status, admission counters, source/output counts, adapter rejections, monotonic processing duration, separate source-preflight duration, and output hashes. |
| `_SUCCESS` | Present only when the completed replay report passes. |

Outputs are built in a temporary directory and published together. A completed
replay containing data-quality failures publishes diagnostics without
`_SUCCESS`; a fatal source/format error does not publish a completed output
directory. The validator reports at most 20 ordinary diagnostics while retaining
full counters; fatal errors can add one final diagnostic. Metadata objects are
limited to 4 MiB and canonical JSONL records to 1 MiB in the standalone reader.

Wall-clock processing duration is distinct from the logical event/availability
clock and does not establish delivery latency or sustained throughput. No
forecast lead-time or real-time performance claim follows from these reports.

## Python interfaces and verification

- `drone_swarm.observations`: immutable validated `Observation`,
  `Observation.from_dict()`, and allowlisted `observation_from_signal()`.
- `drone_swarm.replay_frames`: `FrameAssembler.push()`, `advance()`, `finish()`,
  and bounded `summary()` diagnostics for causal admission.
- `drone_swarm.replay`: `ReplayOptions`, `replay_run()`, and
  `validate_observation_file()` for artifacts and CLI integration.

The tests exercise field isolation, timing modes, covariance and numeric
validation, inclusive deadlines, immediate release and superseding gaps,
duplicate conflicts, missing/late data, reset invalidation, capacity, bounded
storage, and source/output integration. Run the repository test and lint gates
as described in the main README. Mathematical feature integration and sustained
performance qualification remain separate subsequent work.
