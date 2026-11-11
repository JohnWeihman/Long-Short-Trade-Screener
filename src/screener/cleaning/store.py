"""Where cleaned tables live. Everything reads and writes through here, so
moving to partitioned Parquet at full scale is a change in this file only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from screener.data.cache import Cache


def table_path(cache: Cache, name: str, area: str = "clean") -> Path:
    """area is the folder under the data root: 'clean' (Stage 2) or 'scores' (Stage 3)."""
    return cache.root / area / f"{name}.parquet"


def write_table(cache: Cache, name: str, frame: pd.DataFrame, area: str = "clean") -> Path:
    path = table_path(cache, name, area)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def read_table(cache: Cache, name: str, area: str = "clean") -> pd.DataFrame:
    path = table_path(cache, name, area)
    if not path.exists():
        module = "screener.cleaning.build" if area == "clean" else "screener.scoring.build"
        raise FileNotFoundError(f"{path} does not exist. Run: python -m {module}")
    return pd.read_parquet(path)
