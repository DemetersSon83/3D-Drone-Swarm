# Design notes

## Activation semantics

The default model uses staged, simultaneous-style activation:

1. Every drone observes the world at tick `t` and caches `S_t`.
2. Every drone selects an action `A_t` using only cached state and the still-unmoved world.
3. Every drone computes a proposed next position and velocity.
4. Every drone commits movement.
5. Every drone observes `S_{t+1}` and appends a transition record.

This avoids order artifacts in the MDP log. The alternate `activation="random"` mode is still provided for experiments where asynchronous behavior is desired.

## State definition

`DroneState` includes:

- position vector `(x, y, z)`
- velocity vector `(vx, vy, vz)`
- scalar speed
- neighbor count within the perception radius
- nearest-neighbor distance
- local neighbor centroid
- local average neighbor velocity
- optional target vector
- optional battery value and mode label

## Action definition

`DroneAction` currently uses a continuous acceleration vector `(ax, ay, az)`. The boids policy stores interpretable components for cohesion, alignment, separation, target seeking, and boundary avoidance.

## Transition export

The simulator keeps a nested in-memory `Transition` log and provides two export forms:

- JSON Lines: nested, faithful `S, A, S'` objects
- CSV/Parquet: flattened scalar columns for analytics and machine-learning pipelines
