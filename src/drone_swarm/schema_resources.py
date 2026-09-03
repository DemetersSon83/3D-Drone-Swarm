"""Access the machine-readable JSON schemas shipped with the package."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

SchemaName = Literal["scenario", "artifact-manifest", "dataset-catalog"]

_SCHEMA_FILES: dict[SchemaName, str] = {
    "scenario": "scenario.schema.json",
    "artifact-manifest": "artifact-manifest.schema.json",
    "dataset-catalog": "dataset-catalog.schema.json",
}


def load_schema(name: SchemaName) -> dict[str, Any]:
    """Load a bundled JSON schema as a mapping."""

    resource = files("drone_swarm").joinpath("schemas", _SCHEMA_FILES[name])
    value = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(value, dict):  # pragma: no cover - package corruption guard
        raise ValueError(f"bundled schema is not an object: {name}")
    return value


def write_schema(name: SchemaName, output: str | Path) -> Path:
    """Write a bundled schema to *output* and return the resulting path."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(load_schema(name), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
