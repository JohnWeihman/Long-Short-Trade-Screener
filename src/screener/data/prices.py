"""Daily price history. The only module allowed to import yfinance.

To swap the data source, write another class with the PriceSource shape and
pass it to fetch_prices; nothing else in the project needs to change.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from screener.data.cache import Cache, FetchRecord, utc_now

PRICE_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",  # split-adjusted, NOT dividend-adjusted
    "adj_close",  # split- and dividend-adjusted
    "volume",
    "dividends",
    "splits",
]


class PriceDataError(RuntimeError):
    pass


class PriceSource(Protocol):
    name: str

    def history(self, ticker: str) -> pd.DataFrame:
        """Full daily history with exactly PRICE_COLUMNS, sorted by date."""
        ...


_YF_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "splits",
}


def normalize_yfinance(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance Ticker.history(auto_adjust=False) frame -> PRICE_COLUMNS."""
    if raw is None or raw.empty:
        raise PriceDataError(f"No price data returned for {ticker}")
    missing = [c for c in ("Open", "High", "Low", "Close", "Volume") if c not in raw.columns]
    if missing:
        raise PriceDataError(f"{ticker}: price data is missing columns {missing}")

    frame = raw.rename(columns=_YF_RENAME)
    dates = pd.DatetimeIndex(raw.index)
    if dates.tz is not None:
        dates = dates.tz_localize(None)  # keep the exchange-local calendar date
    frame = frame.reset_index(drop=True)
    frame["date"] = dates.normalize()
    frame["ticker"] = ticker.upper()
    if "adj_close" not in frame.columns:
        frame["adj_close"] = float("nan")
    for column in ("dividends", "splits"):
        if column not in frame.columns:
            frame[column] = 0.0

    frame = frame.dropna(subset=["close"])
    frame = frame.drop_duplicates(subset="date", keep="last").sort_values("date")
    return frame[PRICE_COLUMNS].reset_index(drop=True)


class YFinancePriceSource:
    name = "yfinance"

    def history(self, ticker: str) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False, actions=True)
        return normalize_yfinance(raw, ticker)


def fetch_prices(
    source: PriceSource, cache: Cache, ticker: str, refresh: bool = False
) -> FetchRecord:
    """Save a ticker's full daily history as Parquet unless already cached."""
    path = cache.prices_path(source.name, ticker)
    kind = f"prices/{source.name}"
    if path.exists() and not refresh:
        return FetchRecord(kind, ticker, "", str(path), "cached", path.stat().st_size, utc_now())
    frame = source.history(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    detail = f"{len(frame)} rows, {frame['date'].min():%Y-%m-%d} to {frame['date'].max():%Y-%m-%d}"
    return FetchRecord(kind, ticker, "", str(path), "downloaded", path.stat().st_size, utc_now(), detail)


def load_prices(cache: Cache, ticker: str, source_name: str = "yfinance") -> pd.DataFrame:
    return pd.read_parquet(cache.prices_path(source_name, ticker))
