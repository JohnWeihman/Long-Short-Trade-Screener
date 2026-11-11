"""Ticker -> CIK mapping from the SEC's company_tickers.json."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from screener.data.cache import BytesFetcher, Cache, FetchRecord, cik10, fetch_to_cache, read_gz_json

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


class UnknownTickerError(KeyError):
    pass


def normalize_ticker(ticker: str) -> str:
    """SEC and yfinance both write share classes with a dash (BRK-B)."""
    return ticker.strip().upper().replace(".", "-").replace("/", "-")


def fetch_ticker_map(client: BytesFetcher, cache: Cache, refresh: bool = False) -> FetchRecord:
    return fetch_to_cache(
        client,
        TICKER_MAP_URL,
        cache.ticker_map_path,
        source="edgar/company_tickers",
        key="all",
        refresh=refresh,
    )


def parse_ticker_map(raw: dict[str, Any]) -> pd.DataFrame:
    """SEC JSON ({'0': {'cik_str': 320193, 'ticker': 'AAPL', 'title': ...}, ...})
    -> one row per ticker with columns ticker, cik, cik10, name."""
    rows = [
        {
            "ticker": normalize_ticker(str(entry["ticker"])),
            "cik": int(entry["cik_str"]),
            "cik10": cik10(entry["cik_str"]),
            "name": str(entry["title"]),
        }
        for entry in raw.values()
    ]
    frame = pd.DataFrame(rows, columns=["ticker", "cik", "cik10", "name"])
    # The SEC file is ordered by market cap; keep the first row if a ticker repeats.
    return frame.drop_duplicates(subset="ticker", keep="first").reset_index(drop=True)


def load_ticker_map(cache: Cache) -> pd.DataFrame:
    if not cache.ticker_map_path.exists():
        raise FileNotFoundError(
            f"{cache.ticker_map_path} is not cached yet. Run: python -m screener.data.pull"
        )
    return parse_ticker_map(read_gz_json(cache.ticker_map_path))


def resolve_tickers(ticker_map: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Rows of ticker_map for the given tickers, in the order given.

    Raises UnknownTickerError listing every ticker the SEC file does not have.
    """
    wanted = [normalize_ticker(t) for t in tickers]
    indexed = ticker_map.set_index("ticker", drop=False)
    missing = [t for t in wanted if t not in indexed.index]
    if missing:
        raise UnknownTickerError(f"Not in SEC company_tickers.json: {', '.join(missing)}")
    return indexed.loc[wanted].reset_index(drop=True)


def split_known(ticker_map: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """(rows for the tickers the SEC file has, tickers it does not have).

    A ticker goes missing when a company is acquired, delists or renames.
    """
    wanted = [normalize_ticker(t) for t in tickers]
    known = set(ticker_map["ticker"])
    missing = [t for t in wanted if t not in known]
    return resolve_tickers(ticker_map, [t for t in wanted if t in known]), missing


def read_universe(path: Path) -> list[str]:
    """Tickers from a CSV with a 'ticker' column, de-duplicated, order kept."""
    frame = pd.read_csv(path, dtype=str, comment="#")
    if "ticker" not in frame.columns:
        raise ValueError(f"{path} needs a 'ticker' column")
    tickers = [normalize_ticker(t) for t in frame["ticker"].dropna() if t.strip()]
    return list(dict.fromkeys(tickers))
