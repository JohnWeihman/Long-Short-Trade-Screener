"""Where cleaned tables live. Everything reads and writes through here, so
moving to partitioned Parquet at full scale is a change in this file only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from screener.data.cache import Cache


def table_path(cache: Cache, name: str) -> Path:
    return cache.root / "clean" / f"{name}.parquet"


def write_table(cache: Cache, name: str, frame: pd.DataFrame) -> Path:
    path = table_path(cache, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def read_table(cache: Cache, name: str) -> pd.DataFrame:
    path = table_path(cache, name)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run: python -m screener.cleaning.build")
    return pd.read_parquet(path)
