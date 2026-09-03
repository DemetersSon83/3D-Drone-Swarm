"""Loading helpers for pandas, PyArrow, Polars, and DuckDB workflows."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


class DroneSwarmDataset:
    """A filesystem-backed handle to a cataloged swarm dataset."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        catalog_path = self.root / "dataset_catalog.json"
        if not catalog_path.is_file():
            raise FileNotFoundError(
                f"dataset catalog not found: {catalog_path}; run `drone-swarm catalog` first"
            )
        with catalog_path.open(encoding="utf-8") as file_obj:
            value = json.load(file_obj)
        if not isinstance(value, dict):
            raise ValueError(f"dataset catalog must be a JSON object: {catalog_path}")
        self.catalog: dict[str, Any] = value
        self._manifest_cache: Any = None

    def manifest_path(self) -> Path:
        manifest = self.catalog.get("manifest", {})
        parquet = manifest.get("parquet") if isinstance(manifest, dict) else None
        return self.root / (str(parquet) if parquet else "manifest.csv")

    def manifest_pandas(self, *, refresh: bool = False) -> Any:
        import pandas as pd

        if self._manifest_cache is not None and not refresh:
            return self._manifest_cache.copy()
        path = self.manifest_path()
        if not path.is_file():
            raise FileNotFoundError(f"manifest not found: {path}")
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        self._manifest_cache = frame
        return frame.copy()

    def table_paths(
        self,
        table: str,
        *,
        scenario_ids: Iterable[str] | None = None,
        seeds: Iterable[int] | None = None,
        require_exists: bool = True,
    ) -> list[Path]:
        if table not in self.catalog.get("tables", {}):
            raise KeyError(f"table is not present in the dataset catalog: {table}")
        manifest = self.manifest_pandas()
        if scenario_ids is not None:
            manifest = manifest[manifest["scenario_id"].isin(set(scenario_ids))]
        if seeds is not None:
            manifest = manifest[manifest["base_seed"].isin(set(int(value) for value in seeds))]
        column = f"{table}_path"
        if column not in manifest.columns:
            raise KeyError(f"manifest does not contain a path column for table: {table}")
        values = [str(value) for value in manifest[column].dropna().tolist() if str(value)]
        paths = list(dict.fromkeys(self.root / value for value in values))
        if require_exists:
            missing = [path for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "cataloged table files are missing: " + ", ".join(str(path) for path in missing)
                )
        return paths

    def read_pandas(
        self,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        scenario_ids: Iterable[str] | None = None,
        seeds: Iterable[int] | None = None,
    ) -> Any:
        import pandas as pd

        frames = []
        for path in self.table_paths(table, scenario_ids=scenario_ids, seeds=seeds):
            if path.suffix == ".parquet":
                frames.append(pd.read_parquet(path, columns=columns))
            elif path.suffix == ".csv":
                frames.append(pd.read_csv(path, usecols=columns))
            else:
                raise ValueError(f"pandas loader does not support: {path}")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)

    def arrow_dataset(
        self,
        table: str,
        *,
        scenario_ids: Iterable[str] | None = None,
        seeds: Iterable[int] | None = None,
    ) -> Any:
        import pyarrow.dataset as ds

        paths = self.table_paths(table, scenario_ids=scenario_ids, seeds=seeds)
        if not paths:
            raise FileNotFoundError(f"no files found for table: {table}")
        if {path.suffix for path in paths} != {".parquet"}:
            raise ValueError("arrow_dataset requires all selected files to be Parquet")
        return ds.dataset([str(path) for path in paths], format="parquet")

    def polars_lazy(
        self,
        table: str,
        *,
        scenario_ids: Iterable[str] | None = None,
        seeds: Iterable[int] | None = None,
    ) -> Any:
        import polars as pl

        paths = self.table_paths(table, scenario_ids=scenario_ids, seeds=seeds)
        if not paths:
            raise FileNotFoundError(f"no files found for table: {table}")
        suffixes = {path.suffix for path in paths}
        values = [str(path) for path in paths]
        if suffixes == {".parquet"}:
            return pl.scan_parquet(values)
        if suffixes == {".csv"}:
            return pl.scan_csv(values)
        raise ValueError("polars_lazy requires a uniform set of Parquet or CSV inputs")

    def register_duckdb(self, connection: Any, *, prefix: str = "") -> None:
        """Register manifest and table views in an existing DuckDB connection."""

        safe_prefix = "".join(char for char in prefix if char.isalnum() or char == "_")
        if safe_prefix and safe_prefix[0].isdigit():
            safe_prefix = "_" + safe_prefix
        manifest_path = self.manifest_path().as_posix().replace("'", "''")
        if self.manifest_path().suffix == ".parquet":
            connection.execute(
                f"CREATE OR REPLACE VIEW {safe_prefix}manifest AS "
                f"SELECT * FROM read_parquet('{manifest_path}')"
            )
        else:
            connection.execute(
                f"CREATE OR REPLACE VIEW {safe_prefix}manifest AS "
                f"SELECT * FROM read_csv_auto('{manifest_path}', header=true)"
            )
        for table in self.catalog.get("tables", {}):
            paths = self.table_paths(str(table))
            parquet = [
                path.as_posix().replace("'", "''") for path in paths if path.suffix == ".parquet"
            ]
            csv_paths = [
                path.as_posix().replace("'", "''") for path in paths if path.suffix == ".csv"
            ]
            if parquet and len(parquet) == len(paths):
                list_literal = "[" + ",".join(f"'{path}'" for path in parquet) + "]"
                source = f"read_parquet({list_literal}, union_by_name=true, filename=true)"
            elif csv_paths and len(csv_paths) == len(paths):
                list_literal = "[" + ",".join(f"'{path}'" for path in csv_paths) + "]"
                source = (
                    f"read_csv_auto({list_literal}, union_by_name=true, header=true, filename=true)"
                )
            else:
                continue
            connection.execute(
                f"CREATE OR REPLACE VIEW {safe_prefix}{table} AS SELECT * FROM {source}"
            )
