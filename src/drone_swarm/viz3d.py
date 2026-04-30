"""Optional 3D plotting helpers for swarm trajectories."""

from __future__ import annotations

from pathlib import Path


def plot_transition_trajectories(
    transitions_df,
    *,
    output_path: str | Path | None = None,
    max_agents: int | None = None,
):  # type: ignore[no-untyped-def]
    """Plot agent trajectories from a flattened transition DataFrame.

    The DataFrame should contain the columns created by
    ``drone_swarm.mdp.transition_to_row``. The function returns the Matplotlib
    ``Figure`` so callers can further customize or display it.
    """

    import matplotlib.pyplot as plt

    required = {"agent_id", "step", "sp_position_x", "sp_position_y", "sp_position_z"}
    missing = required.difference(transitions_df.columns)
    if missing:
        raise ValueError(f"transitions_df is missing required columns: {sorted(missing)}")

    agent_ids = list(transitions_df["agent_id"].drop_duplicates())
    if max_agents is not None:
        agent_ids = agent_ids[:max_agents]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for agent_id in agent_ids:
        agent_df = transitions_df[transitions_df["agent_id"] == agent_id].sort_values("step")
        ax.plot(
            agent_df["sp_position_x"],
            agent_df["sp_position_y"],
            agent_df["sp_position_z"],
            label=str(agent_id),
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Drone swarm trajectories")

    if max_agents is None or len(agent_ids) <= 12:
        ax.legend(title="agent_id")

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    return fig
