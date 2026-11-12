"""Return statistics for pairs: correlation, betas and hedge ratios.

The hedge ratio makes a pair beta-neutral: for every $1 long, short
beta_long / beta_short dollars, so the two legs have the same exposure to
the benchmark and what is left is the bet between the two companies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A return series whose standard deviation is below this never really moved
# (floating-point noise around a constant), so it has no correlation or beta.
NO_MOVEMENT = 1e-9


def correlation(a: pd.Series, b: pd.Series, min_overlap: int) -> tuple[float, int]:
    """(correlation, weeks both have a return). NaN if the overlap is too short."""
    both = pd.concat([a, b], axis=1).dropna()
    n = len(both)
    if n < min_overlap or both.iloc[:, 0].std() < NO_MOVEMENT or both.iloc[:, 1].std() < NO_MOVEMENT:
        return float("nan"), n
    return float(both.iloc[:, 0].corr(both.iloc[:, 1])), n


def beta(asset: pd.Series, benchmark: pd.Series, min_overlap: int) -> float:
    """Slope of the asset's returns on the benchmark's: covariance / variance."""
    both = pd.concat([asset, benchmark], axis=1).dropna()
    if len(both) < min_overlap:
        return float("nan")
    variance = both.iloc[:, 1].var()
    if np.isnan(variance) or variance < NO_MOVEMENT**2:  # a benchmark that never moves has no slope
        return float("nan")
    return float(both.iloc[:, 0].cov(both.iloc[:, 1]) / variance)


def equal_weight_benchmark(returns: pd.DataFrame) -> pd.Series:
    """Average return across all companies with data each period."""
    return returns.mean(axis=1, skipna=True)


def betas(returns: pd.DataFrame, benchmark: pd.Series, min_overlap: int) -> pd.Series:
    """Beta of every column of returns against the benchmark."""
    return pd.Series({ticker: beta(returns[ticker], benchmark, min_overlap) for ticker in returns.columns})


def hedge_ratio(beta_long: float, beta_short: float) -> float:
    """Dollars to short per dollar long for a beta-neutral pair.

    Undefined (NaN) unless both betas are known and the short's is positive:
    a zero or negative beta cannot hedge anything.
    """
    if np.isnan(beta_long) or np.isnan(beta_short) or beta_short <= 0:
        return float("nan")
    return beta_long / beta_short


def share_ratio(ratio: float, price_long: float, price_short: float) -> float:
    """Shares to short per share long: the dollar hedge ratio converted at current prices."""
    if np.isnan(ratio) or not price_short or np.isnan(price_short) or np.isnan(price_long):
        return float("nan")
    return ratio * price_long / price_short
