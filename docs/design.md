# Design notes

## Activation semantics

The default model uses staged, simultaneous-style activation:

1. Every drone observes the world at tick `t` and caches `S_t`.
2. Every drone selects an action `A_t` using only cached state and the still-unmoved world.
3. Every drone computes a proposed next position and velocity.
4. Every drone commits movement.
5. Every drone observes `S_{t+1}` and appends a transition record.

The Phase I experiment layer inserts explicit observation, communication, policy,
actuator, and environment boundaries into that lifecycle. This avoids order
artifacts and preserves aligned controller-facing and plant-facing interaction
tokens. The alternate `activation="random"` mode remains available as an
asynchrony stress condition.

## State definition

`DroneState` includes:

- position vector `(x, y, z)`
- velocity vector `(vx, vy, vz)`
- scalar speed
- neighbor count within the effective perception radius
- nearest-neighbor distance
- local neighbor centroid
- local average neighbor velocity
- perceived neighbor IDs
- local separation steering summary
- optional target vector
- optional battery value and mode label

## Action definition

`DroneAction` uses a continuous acceleration vector `(ax, ay, az)`. The boids
policy stores interpretable components for cohesion, alignment, separation,
target seeking, and boundary avoidance. A transition stores both commanded and
applied actions so controller changes can be distinguished from actuator faults.

## Transition export

The canonical run artifact is Parquet. CSV and nested JSON Lines are optional.
The experiment runner streams records in bounded batches rather than retaining a
full run in memory. It also emits a compact `agent_signals` projection, run-level
provenance, artifact inventories, quality reports, and dataset-level catalogs.
