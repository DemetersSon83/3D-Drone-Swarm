"""Optional 3D plotting helpers for swarm trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_transition_trajectories(
    transitions_df: Any,
    *,
    output_path: str | Path | None = None,
    max_agents: int | None = None,
) -> Any:
    """Plot agent trajectories from transitions or compact agent signals.

    The function accepts either the canonical transition columns
    ``sp_position_x/y/z`` or the compact projection columns
    ``observed_next_position_x/y/z``.
    """

    import matplotlib.pyplot as plt

    candidates = (
        ("sp_position_x", "sp_position_y", "sp_position_z"),
        (
            "observed_next_position_x",
            "observed_next_position_y",
            "observed_next_position_z",
        ),
    )
    coordinates: tuple[str, str, str] | None = None
    for candidate in candidates:
        if set(candidate).issubset(transitions_df.columns):
            coordinates = candidate
            break
    required = {"agent_id", "step"}
    missing = required.difference(transitions_df.columns)
    if missing or coordinates is None:
        missing_values = sorted(missing)
        if coordinates is None:
            missing_values.append("one supported x/y/z position triplet")
        raise ValueError(f"table is missing required columns: {missing_values}")

    agent_ids = list(transitions_df["agent_id"].drop_duplicates())
    if max_agents is not None:
        agent_ids = agent_ids[:max_agents]

    fig = plt.figure()
    axis = fig.add_subplot(111, projection="3d")
    x_name, y_name, z_name = coordinates
    for agent_id in agent_ids:
        agent_df = transitions_df[transitions_df["agent_id"] == agent_id].sort_values("step")
        axis.plot(
            agent_df[x_name],
            agent_df[y_name],
            agent_df[z_name],
            label=str(agent_id),
        )

    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title("Drone swarm trajectories")

    if max_agents is None or len(agent_ids) <= 12:
        axis.legend(title="agent_id")

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")

    return fig
