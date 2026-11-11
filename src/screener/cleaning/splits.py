"""Put every share count and per-share figure on today's share basis.

EDGAR values are as reported: a 10-Q filed before a 10-for-1 split shows
one tenth of the shares (and ten times the EPS) that a filing after the
split shows for the very same period. Prices from the price source are
already adjusted for every split up to today, so share counts must be too
before the two are combined, and before quarters are derived (a full-year
share count restated for a split minus a nine-month count that was not is
nonsense).

Rule: a value is multiplied (shares) or divided (per-share) by the ratios
of all splits that took effect AFTER the value was stated:
  - us-gaap facts: after the filing date. Financial statements issued after
    a split restate every period shown for it.
  - dei cover-page facts: after the date of the count itself. The cover
    page is a plain count on a stated day, never restated.
Which facts are affected is decided by unit alone ('shares', 'USD/shares').
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.data.cache import Cache

SPLIT_COLUMNS = ["cik10", "split_date", "ratio"]
SHARE_UNIT = "shares"
PER_SHARE_UNIT = "USD/shares"


def load_splits(cache: Cache, tickers: dict[str, str], source: str = "yfinance") -> pd.DataFrame:
    """Split history from the cached price files.

    tickers maps cik10 -> the ticker its prices are stored under. ratio is
    new shares per old share (10.0 for a 10-for-1 split, 0.1 for a 1-for-10
    reverse split). Companies with no price file simply have no splits here.
    """
    frames = []
    for cik10, ticker in tickers.items():
        path = cache.prices_path(source, ticker)
        if not path.exists():
            continue
        prices = pd.read_parquet(path, columns=["date", "splits"])
        splits = prices[(prices["splits"] > 0) & (prices["splits"] != 1)]
        frames.append(pd.DataFrame({
            "cik10": cik10,
            "split_date": pd.to_datetime(splits["date"]).astype("datetime64[ns]"),
            "ratio": splits["splits"].astype("float64"),
        }))
    if not frames:
        return pd.DataFrame(columns=SPLIT_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["cik10", "split_date"], kind="stable").reset_index(drop=True)


def split_factor(cik10: pd.Series, stated_on: pd.Series, splits: pd.DataFrame) -> pd.Series:
    """Product of the ratios of each company's splits dated after stated_on.

    A value stated on the split date itself is treated as already post-split.
    """
    factor = pd.Series(1.0, index=cik10.index)
    if splits.empty:
        return factor
    stated = pd.to_datetime(stated_on).astype("datetime64[ns]")
    for company, group in splits.groupby("cik10", sort=False):
        rows = (cik10 == company).to_numpy()
        if not rows.any():
            continue
        group = group.sort_values("split_date", kind="stable")
        dates = group["split_date"].to_numpy(dtype="datetime64[ns]")
        # remaining[i] = product of ratios of splits i, i+1, ...; remaining[n] = 1
        remaining = np.append(np.cumprod(group["ratio"].to_numpy()[::-1])[::-1], 1.0)
        first_later = np.searchsorted(dates, stated[rows].to_numpy(dtype="datetime64[ns]"), side="right")
        factor[rows] = remaining[first_later]
    return factor


def adjust_for_splits(raw: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """facts_raw with share and per-share values restated to today's basis.

    Adds val_reported (the value as filed) and split_factor (what shares
    were multiplied by; 1.0 when nothing changed). Other units are untouched.
    """
    stated_on = raw["filed"].where(raw["taxonomy"] != "dei", raw["end"])
    factor = split_factor(raw["cik10"], stated_on, splits)
    is_shares = raw["unit"] == SHARE_UNIT
    is_per_share = raw["unit"] == PER_SHARE_UNIT
    factor = factor.where(is_shares | is_per_share, 1.0)

    out = raw.copy()
    out["val_reported"] = raw["val"]
    out["split_factor"] = factor
    out["val"] = np.where(is_per_share, raw["val"] / factor, raw["val"] * factor)
    return out
