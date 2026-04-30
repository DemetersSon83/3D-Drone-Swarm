"""Input/output helpers for transition logs and telemetry."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from drone_swarm.mdp import Transition, transition_to_row


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for *path* and return a ``Path`` object."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def transitions_to_rows(transitions: Iterable[Transition]) -> list[dict[str, Any]]:
    """Return flattened rows for a collection of transitions."""

    return [transition_to_row(transition) for transition in transitions]


def transitions_to_dataframe(transitions: Iterable[Transition]):  # type: ignore[no-untyped-def]
    """Convert transitions to a pandas DataFrame."""

    import pandas as pd

    return pd.DataFrame(transitions_to_rows(transitions))


def write_transitions_jsonl(transitions: Iterable[Transition], path: str | Path) -> None:
    """Write nested transition objects to a JSON Lines file."""

    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as file_obj:
        for transition in transitions:
            file_obj.write(json.dumps(transition.to_dict(), sort_keys=True) + "\n")


def write_transitions_csv(transitions: Iterable[Transition], path: str | Path) -> None:
    """Write flattened transition rows to CSV."""

    output_path = ensure_parent_dir(path)
    df = transitions_to_dataframe(transitions)
    df.to_csv(output_path, index=False)


def write_transitions_parquet(transitions: Iterable[Transition], path: str | Path) -> None:
    """Write flattened transition rows to Parquet.

    Install the ``parquet`` extra for PyArrow support:
    ``python -m pip install -e '.[parquet]'``.
    """

    output_path = ensure_parent_dir(path)
    df = transitions_to_dataframe(transitions)
    df.to_parquet(output_path, index=False)
