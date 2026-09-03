"""Model- and swarm-level metric helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from drone_swarm.physics import norm


class AgentLike(Protocol):
    unique_id: int
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]


type MappingLike = Mapping[int, Sequence[int]]


def _agent_list(agents: Iterable[AgentLike]) -> list[AgentLike]:
    return list(agents)


def mean_speed(agents: Iterable[AgentLike]) -> float:
    """Return mean agent speed for objects with a ``velocity`` attribute."""

    speeds = [norm(agent.velocity) for agent in agents]
    return float(np.mean(speeds)) if speeds else 0.0


def min_pairwise_distance(agents: Iterable[AgentLike]) -> float | None:
    """Return the minimum Euclidean distance between any pair of agents."""

    distances: list[float] = []
    for left, right in combinations(agents, 2):
        distances.append(float(np.linalg.norm(left.position - right.position)))
    if not distances:
        return None
    return min(distances)


def collision_count(agents: Iterable[AgentLike], *, collision_radius: float) -> int:
    """Count pairwise proximity violations under ``collision_radius``."""

    if collision_radius < 0:
        raise ValueError("collision_radius must be non-negative")

    count = 0
    for left, right in combinations(agents, 2):
        if float(np.linalg.norm(left.position - right.position)) <= collision_radius:
            count += 1
    return count


def swarm_centroid(agents: Iterable[AgentLike]) -> tuple[float, float, float] | None:
    """Return the centroid of agent positions, if at least one agent exists."""

    positions = [agent.position for agent in agents]
    if not positions:
        return None
    centroid = np.mean(np.asarray(positions, dtype=float), axis=0)
    return (float(centroid[0]), float(centroid[1]), float(centroid[2]))


def polarization(agents: Iterable[AgentLike], *, eps: float = 1e-12) -> float:
    """Return the norm of the mean unit velocity, in ``[0, 1]``."""

    directions: list[np.ndarray] = []
    for agent in agents:
        velocity = np.asarray(agent.velocity, dtype=float)
        speed = np.linalg.norm(velocity)
        if speed > eps:
            directions.append(velocity / speed)
    if not directions:
        return 0.0
    return float(np.linalg.norm(np.mean(np.asarray(directions), axis=0)))


def radius_of_gyration(agents: Iterable[AgentLike]) -> float:
    """Return root-mean-square distance from the swarm centroid."""

    positions = np.asarray([agent.position for agent in agents], dtype=float)
    if positions.size == 0:
        return 0.0
    centroid = positions.mean(axis=0)
    squared = np.sum((positions - centroid) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared)))


def position_shape_metrics(agents: Iterable[AgentLike]) -> dict[str, float]:
    """Return position-covariance eigenvalues and a simple anisotropy index."""

    positions = np.asarray([agent.position for agent in agents], dtype=float)
    if len(positions) < 2:
        return {
            "position_eigenvalue_1": 0.0,
            "position_eigenvalue_2": 0.0,
            "position_eigenvalue_3": 0.0,
            "position_anisotropy": 0.0,
        }
    centered = positions - positions.mean(axis=0)
    covariance = centered.T @ centered / len(positions)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    total = float(np.sum(eigenvalues))
    anisotropy = 0.0 if total <= 0 else float((eigenvalues[0] - eigenvalues[-1]) / total)
    return {
        "position_eigenvalue_1": float(eigenvalues[0]),
        "position_eigenvalue_2": float(eigenvalues[1]),
        "position_eigenvalue_3": float(eigenvalues[2]),
        "position_anisotropy": anisotropy,
    }


def _graph_components(adjacency: np.ndarray) -> list[list[int]]:
    n_nodes = adjacency.shape[0]
    remaining = set(range(n_nodes))
    components: list[list[int]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            node = stack.pop()
            neighbors = np.flatnonzero(adjacency[node]).tolist()
            for neighbor in neighbors:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(component)
    return components


def graph_metrics_from_neighbor_ids(
    agent_ids: Sequence[int],
    neighbor_ids: MappingLike,
) -> dict[str, float | int]:
    """Compute undirected interaction-graph summaries from realized neighbor ids.

    Directed observations are symmetrized: an undirected edge exists when either
    endpoint reports the other.  This keeps the summary meaningful under
    receiver-specific communication dropout.
    """

    count = len(agent_ids)
    if count == 0:
        return {
            "interaction_mean_degree": 0.0,
            "interaction_component_count": 0,
            "interaction_largest_component_fraction": 0.0,
            "interaction_algebraic_connectivity": 0.0,
        }

    index_by_id = {int(agent_id): index for index, agent_id in enumerate(agent_ids)}
    adjacency = np.zeros((count, count), dtype=float)
    for source_id, targets in neighbor_ids.items():
        source_index = index_by_id.get(int(source_id))
        if source_index is None:
            continue
        for target_id in targets:
            target_index = index_by_id.get(int(target_id))
            if target_index is None or target_index == source_index:
                continue
            adjacency[source_index, target_index] = 1.0
            adjacency[target_index, source_index] = 1.0

    degrees = adjacency.sum(axis=1)
    components = _graph_components(adjacency)
    if count <= 1:
        algebraic_connectivity = 0.0
    else:
        laplacian = np.diag(degrees) - adjacency
        eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
        algebraic_connectivity = float(max(0.0, eigenvalues[1]))

    return {
        "interaction_mean_degree": float(np.mean(degrees)),
        "interaction_component_count": len(components),
        "interaction_largest_component_fraction": max(map(len, components)) / count,
        "interaction_algebraic_connectivity": algebraic_connectivity,
    }


def swarm_metrics_snapshot(
    agents: Iterable[AgentLike],
    *,
    tick: int,
    transition_count: int,
    collision_radius: float,
    realized_neighbor_ids: MappingLike | None = None,
) -> dict[str, Any]:
    """Return one analysis-ready row of conventional swarm diagnostics."""

    agent_values = _agent_list(agents)
    centroid = swarm_centroid(agent_values)
    row: dict[str, Any] = {
        "tick": tick,
        "n_drones": len(agent_values),
        "transition_count": transition_count,
        "mean_speed": mean_speed(agent_values),
        "min_pairwise_distance": min_pairwise_distance(agent_values),
        "collision_count": collision_count(agent_values, collision_radius=collision_radius),
        "centroid_x": None if centroid is None else centroid[0],
        "centroid_y": None if centroid is None else centroid[1],
        "centroid_z": None if centroid is None else centroid[2],
        "polarization": polarization(agent_values),
        "radius_of_gyration": radius_of_gyration(agent_values),
    }
    row.update(position_shape_metrics(agent_values))
    if realized_neighbor_ids is not None:
        row.update(
            graph_metrics_from_neighbor_ids(
                [int(agent.unique_id) for agent in agent_values],
                realized_neighbor_ids,
            )
        )
    return row
