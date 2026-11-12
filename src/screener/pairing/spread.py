"""The hedged spread of a pair through time.

Each day's spread return is the long's return minus the hedge ratio times
the short's return: what $1 long and `hedge_ratio` dollars short would have
made that day. The spread series is the running total of those returns,
starting at zero (so it assumes the position is reset to the hedge ratio
every day).

The hedge ratio is today's estimate applied to the whole history. That is
right for "what has this pair, as I would put it on now, been doing", and
is not a backtest of what would have been traded at the time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPREAD_COLUMNS = ["pair_id", "date", "long_return", "short_return", "spread_return", "spread"]
LOOKBACKS = {"spread_chg_1m": 21, "spread_chg_3m": 63, "spread_chg_12m": 252}


def daily_returns(prices: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.Series:
    """Day-over-day returns with dividends reinvested, indexed by date."""
    series = prices.set_index("date")["adj_close"].dropna()
    if as_of is not None:
        series = series[series.index <= as_of]
    return series.pct_change().iloc[1:]


def spread_series(
    pair_id: str, long_returns: pd.Series, short_returns: pd.Series, ratio: float, history_days: int
) -> pd.DataFrame:
    """Daily hedged spread for the last history_days days both stocks traded."""
    both = pd.concat([long_returns, short_returns], axis=1, keys=["long_return", "short_return"]).dropna()
    both = both.tail(history_days)
    if both.empty or np.isnan(ratio):
        return pd.DataFrame(columns=SPREAD_COLUMNS)
    both["spread_return"] = both["long_return"] - ratio * both["short_return"]
    both["spread"] = both["spread_return"].cumsum()
    both = both.rename_axis("date").reset_index()
    both.insert(0, "pair_id", pair_id)
    return both[SPREAD_COLUMNS]


def spread_stats(spread: pd.DataFrame, zscore_window: int) -> dict:
    """Where the spread stands now: z-score against its recent history, and
    how much it has moved over the last 1, 3 and 12 months.

    A positive z-score means the long has already outperformed the hedged
    short by more than usual over the window.
    """
    out = {"spread_z": float("nan"), **{name: float("nan") for name in LOOKBACKS}}
    if spread.empty:
        return out
    level = spread["spread"]
    window = level.tail(zscore_window)
    if len(window) >= zscore_window // 2 and window.std() > 0:
        out["spread_z"] = float((level.iloc[-1] - window.mean()) / window.std())
    for name, days in LOOKBACKS.items():
        if len(spread) >= days:
            out[name] = float(spread["spread_return"].tail(days).sum())
    return out
