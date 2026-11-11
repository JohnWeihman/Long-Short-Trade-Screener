"""Local raw-data cache: where files live, and fetch-only-if-missing logic.

Everything from EDGAR is keyed by 10-digit zero-padded CIK and stored
byte-for-byte (gzipped) under the SEC's own file names.
"""

from __future__ import annotations

import gzip
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

import pandas as pd

from screener.data.http import NotFoundError


class BytesFetcher(Protocol):
    def get(self, url: str) -> bytes: ...


def cik10(cik: int | str) -> str:
    """320193 -> '0000320193'."""
    digits = str(cik).strip().upper().removeprefix("CIK")
    if not digits.isdigit() or len(digits) > 10:
        raise ValueError(f"Not a valid CIK: {cik!r}")
    return digits.zfill(10)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


@dataclass(frozen=True)
class FetchRecord:
    """One row of the fetch manifest."""

    source: str  # e.g. 'edgar/companyfacts', 'prices/yfinance'
    key: str  # CIK10 or ticker
    url: str
    path: str
    status: str  # 'downloaded' | 'cached' | 'not_found' | 'error'
    bytes: int
    at: str  # UTC ISO timestamp of this attempt
    detail: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Cache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @property
    def ticker_map_path(self) -> Path:
        return self.root / "raw" / "edgar" / "company_tickers.json.gz"

    def companyfacts_path(self, cik: int | str) -> Path:
        return self.root / "raw" / "edgar" / "companyfacts" / f"CIK{cik10(cik)}.json.gz"

    def submissions_path(self, file_name: str) -> Path:
        """file_name is the SEC's, e.g. 'CIK0000320193.json'."""
        return self.root / "raw" / "edgar" / "submissions" / f"{_safe_name(file_name)}.gz"

    def prices_path(self, source: str, ticker: str) -> Path:
        name = _safe_name(ticker.upper())
        return self.root / "raw" / "prices" / _safe_name(source) / f"{name}.parquet"

    @property
    def manifest_dir(self) -> Path:
        return self.root / "manifest"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_gz(path: Path, content: bytes) -> None:
    _atomic_write(path, gzip.compress(content))


def read_gz(path: Path) -> bytes:
    return gzip.decompress(Path(path).read_bytes())


def read_gz_json(path: Path) -> Any:
    return json.loads(read_gz(path))


def fetch_to_cache(
    client: BytesFetcher,
    url: str,
    path: Path,
    *,
    source: str,
    key: str,
    refresh: bool = False,
) -> FetchRecord:
    """Download url to path unless it is already cached (or refresh is set)."""
    if path.exists() and not refresh:
        return FetchRecord(source, key, url, str(path), "cached", path.stat().st_size, utc_now())
    try:
        content = client.get(url)
    except NotFoundError as exc:
        return FetchRecord(source, key, url, str(path), "not_found", 0, utc_now(), str(exc))
    write_gz(path, content)
    return FetchRecord(source, key, url, str(path), "downloaded", len(content), utc_now())


def write_manifest(cache: Cache, records: Iterable[FetchRecord]) -> Path | None:
    """Save this run's fetch records as one Parquet file; None if no records."""
    rows = [asdict(r) for r in records]
    if not rows:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = cache.manifest_dir / f"run_{stamp}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path
