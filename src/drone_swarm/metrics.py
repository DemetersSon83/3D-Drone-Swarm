"""Model and swarm metric helpers."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

import numpy as np

from drone_swarm.physics import norm


def mean_speed(agents: Iterable[object]) -> float:
    """Return mean agent speed for objects with a ``velocity`` attribute."""

    speeds = [norm(getattr(agent, "velocity")) for agent in agents]
    return float(np.mean(speeds)) if speeds else 0.0


def min_pairwise_distance(agents: Iterable[object]) -> float | None:
    """Return the minimum Euclidean distance between any pair of agents."""

    distances: list[float] = []
    for left, right in combinations(agents, 2):
        distances.append(float(np.linalg.norm(left.position - right.position)))
    if not distances:
        return None
    return min(distances)


def collision_count(agents: Iterable[object], *, collision_radius: float) -> int:
    """Count pairwise proximity violations under ``collision_radius``."""

    if collision_radius < 0:
        raise ValueError("collision_radius must be non-negative")

    count = 0
    for left, right in combinations(agents, 2):
        if float(np.linalg.norm(left.position - right.position)) <= collision_radius:
            count += 1
    return count


def swarm_centroid(agents: Iterable[object]) -> tuple[float, float, float] | None:
    """Return the centroid of agent positions, if at least one agent exists."""

    positions = [agent.position for agent in agents]
    if not positions:
        return None
    centroid = np.mean(np.asarray(positions, dtype=float), axis=0)
    return (float(centroid[0]), float(centroid[1]), float(centroid[2]))
