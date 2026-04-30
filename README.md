# Drone Swarm Boids

A Mesa-based 3D boids-style agent-based model for drone swarm simulation. The simulator logs per-agent Markov decision process transitions as `(S_t, A_t, S_{t+1})`, including position and velocity vectors for each state.

This repository is intended for research and simulation of coordination, formation dynamics, collision avoidance, and policy evaluation in synthetic 3D environments.

## Why staged activation?

The default activation mode is `staged`, which implements move-all semantics:

```text
observe all agents       -> S_t
decide all actions       -> A_t
compute proposed motion
commit all movement      -> positions/velocities at t+1
observe all agents again -> S_{t+1}
log transitions
```

This produces clean same-tick MDP records and avoids random-order artifacts. An alternate `activation="random"` mode is available for experiments with asynchronous behavior.

## Repository layout

```text
src/drone_swarm/
  model.py       # Mesa model and staged activation loop
  drone.py       # Continuous-space drone agent
  policies.py    # Boids, hold, and random acceleration policies
  mdp.py         # DroneState, DroneAction, Transition schemas
  physics.py     # 3D vector math, kinematics, boundary handling
  metrics.py     # swarm-level metrics
  io.py          # CSV, JSONL, and Parquet export helpers
  viz3d.py       # optional 3D trajectory plotting
examples/
  run_basic_swarm.py
  plot_trajectories.py
tests/
  test_physics.py
  test_mdp_logging.py
  test_model_smoke.py
```

## Install

Mesa 3.5.1 currently requires Python 3.12 or newer, so create a Python 3.12+ environment first.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,viz,parquet]"
```

## Run a simulation

```bash
python examples/run_basic_swarm.py --steps 100 --n-drones 50 --output outputs/transitions.csv
```

The CSV contains one row per agent per tick. For example, with 50 drones and 100 steps, the transition table has 5,000 rows.

To also export nested JSON Lines:

```bash
python examples/run_basic_swarm.py \
  --steps 100 \
  --n-drones 50 \
  --output outputs/transitions.csv \
  --jsonl-output outputs/transitions.jsonl
```

## Plot trajectories

```bash
python examples/plot_trajectories.py outputs/transitions.csv --output outputs/trajectories.png
```

## Minimal usage

```python
from drone_swarm.model import DroneSwarmModel

model = DroneSwarmModel(
    n_drones=25,
    bounds=((0, 100), (0, 100), (0, 50)),
    seed=42,
    activation="staged",
)

model.run_steps(50)
df = model.transitions_dataframe()
print(df[["step", "agent_id", "s_position_x", "s_velocity_x", "a_acceleration_x", "sp_position_x"]].head())
```

## Transition schema

Each `Transition` contains:

```python
Transition(
    episode_id="...",
    step=0,
    agent_id=1,
    state=DroneState(...),       # S_t
    action=DroneAction(...),     # A_t
    next_state=DroneState(...),  # S_{t+1}
    reward=None,
    done=False,
)
```

`DroneState` includes `position=(x, y, z)` and `velocity=(vx, vy, vz)`, plus speed, local neighbor features, optional target vector, optional battery, and mode.

`DroneAction` currently uses continuous acceleration: `acceleration=(ax, ay, az)`. The default boids policy also stores steering components for interpretability.

## Tests

```bash
pytest
```

`test_model_smoke.py` requires Mesa. Pure schema and physics tests can run without Mesa installed when `PYTHONPATH=src` is set.

## Notes

- The model uses Mesa's `mesa.experimental.continuous_space` namespace for arbitrary-dimensional continuous space.
- The project pins `mesa>=3.5,<4.0` because Mesa 4.0 is currently an alpha pre-release with breaking API changes.
- This is a simulation codebase, not a flight-control stack.
