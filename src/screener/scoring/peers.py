"""Peer groups: the curated file, and a check of it against price behaviour.

    python -m screener.scoring.peers              # 3 years of weekly returns
    python -m screener.scoring.peers --weeks 260

The rest of scoring depends only on the (ticker -> peer_group) table, so the
curated file can later be replaced by a data-driven grouping without changes
elsewhere.

The check: true peers should move together beyond what the whole sector
does. Each company's weekly returns have the sector's move taken out, and
the leftover ("residual") returns are correlated. A company whose residual
is more correlated with another group than with its own is reported.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from screener import config
from screener.data.cache import Cache
from screener.data.prices import load_prices
from screener.data.tickers import normalize_ticker, read_universe

DEFAULT_PEER_GROUPS = config.PROJECT_ROOT / "config" / "peer_groups.csv"
MIN_WEEKS = 52
AFFINITY_COLUMNS = ["ticker", "peer_group", "own_corr", "best_group", "best_corr", "agrees", "closest"]


class PeerGroupError(ValueError):
    pass


def load_peer_groups(path: Path = DEFAULT_PEER_GROUPS) -> pd.DataFrame:
    """ticker, peer_group (None when the company is unplaced). One row per ticker."""
    frame = pd.read_csv(path, dtype=str, comment="#", keep_default_na=False)
    if list(frame.columns) != ["ticker", "peer_group"]:
        raise PeerGroupError(f"{path} must have exactly the columns ticker,peer_group")
    frame["ticker"] = frame["ticker"].map(normalize_ticker)
    frame["peer_group"] = frame["peer_group"].str.strip()
    frame["peer_group"] = frame["peer_group"].where(frame["peer_group"] != "", None)
    repeated = frame["ticker"][frame["ticker"].duplicated()].tolist()
    if repeated:
        raise PeerGroupError(f"tickers listed more than once: {', '.join(repeated)}")
    return frame.reset_index(drop=True)


def check_against_universe(groups: pd.DataFrame, universe: list[str]) -> tuple[list[str], list[str]]:
    """(universe tickers missing from the peer file, peer-file tickers not in the universe)."""
    listed = set(groups["ticker"])
    return [t for t in universe if t not in listed], sorted(listed - set(universe))


def weekly_returns(prices: dict[str, pd.DataFrame], weeks: int = 156, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Week-over-week returns (Friday close, dividends reinvested), one column
    per ticker, for the last `weeks` weeks up to as_of."""
    closes = {}
    for ticker, frame in prices.items():
        series = frame.set_index("date")["adj_close"].dropna()
        if as_of is not None:
            series = series[series.index <= as_of]
        closes[ticker] = series.resample("W-FRI").last()
    wide = pd.DataFrame(closes).sort_index()
    returns = wide.pct_change(fill_method=None).iloc[1:]
    return returns.tail(weeks)


def residual_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Each company's return with the sector's move removed.

    The sector is the equal-weighted average of the OTHER companies (so a
    company is never regressed on itself); the residual is the return minus
    beta times that average, fitted over the window.
    """
    out = {}
    for ticker in returns.columns:
        own = returns[ticker]
        sector = returns.drop(columns=ticker).mean(axis=1)
        both = pd.concat([own, sector], axis=1, keys=["own", "sector"]).dropna()
        if len(both) < MIN_WEEKS or both["sector"].var() == 0:
            continue
        beta = both["own"].cov(both["sector"]) / both["sector"].var()
        alpha = both["own"].mean() - beta * both["sector"].mean()
        out[ticker] = both["own"] - alpha - beta * both["sector"]
    return pd.DataFrame(out)


def group_affinity(correlation: pd.DataFrame, groups: pd.DataFrame, closest: int = 3) -> pd.DataFrame:
    """For each company: average correlation with its own group, the group
    it is most correlated with, and its most correlated individual peers.

    `agrees` is True when no other group beats its own; it is None for
    unplaced companies and for groups of one.
    """
    members = groups.dropna(subset=["peer_group"]).groupby("peer_group")["ticker"].apply(list).to_dict()
    assigned = groups.set_index("ticker")["peer_group"].to_dict()
    rows = []
    for ticker in correlation.index:
        others = correlation.loc[ticker].drop(index=ticker).dropna()
        by_group = {
            name: others.reindex([t for t in tickers if t != ticker]).dropna().mean()
            for name, tickers in members.items()
        }
        by_group = {name: value for name, value in by_group.items() if not np.isnan(value)}
        own_group = assigned.get(ticker)
        own_group = None if pd.isna(own_group) else own_group  # unplaced
        own = by_group.get(own_group) if own_group else None
        best_group = max(by_group, key=by_group.get) if by_group else None
        rows.append({
            "ticker": ticker,
            "peer_group": own_group,
            "own_corr": own,
            "best_group": best_group,
            "best_corr": by_group.get(best_group) if best_group else None,
            "agrees": None if own is None else bool(best_group == own_group),
            "closest": ", ".join(f"{t} {v:.2f}" for t, v in others.nlargest(closest).items()),
        })
    frame = pd.DataFrame(rows, columns=AFFINITY_COLUMNS)
    for column in ("peer_group", "agrees"):  # keep None as None, not NaN
        frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weeks", type=int, default=156)
    args = parser.parse_args(argv)

    cache = Cache(config.data_dir())
    groups = load_peer_groups()
    missing, extra = check_against_universe(groups, read_universe(config.DEFAULT_UNIVERSE))
    if missing:
        print(f"In the universe but not in the peer file: {', '.join(missing)}")
    if extra:
        print(f"In the peer file but not in the universe: {', '.join(extra)}")

    prices = {t: load_prices(cache, t) for t in groups["ticker"] if cache.prices_path("yfinance", t).exists()}
    returns = weekly_returns(prices, args.weeks)
    residuals = residual_returns(returns)
    affinity = group_affinity(residuals.corr(min_periods=MIN_WEEKS), groups)

    print(f"Weekly returns {returns.index[0]:%Y-%m-%d} to {returns.index[-1]:%Y-%m-%d} "
          f"({len(returns)} weeks), sector move removed\n")
    print(f"{'ticker':<6} {'assigned group':<27} {'own':>5}  {'most correlated group':<27} {'corr':>5}  closest companies")
    order = affinity.sort_values(["peer_group", "ticker"], na_position="last")
    for row in order.itertuples():
        own = "  -  " if row.own_corr is None or pd.isna(row.own_corr) else f"{row.own_corr:5.2f}"
        flag = "" if row.agrees is not False else "   <-- differs"
        group = "(unplaced)" if row.peer_group is None or pd.isna(row.peer_group) else row.peer_group
        print(f"{row.ticker:<6} {group:<27} {own}  "
              f"{str(row.best_group):<27} {row.best_corr:5.2f}  {row.closest}{flag}")

    placed = affinity[affinity["agrees"].notna()]
    print(f"\n{int(placed['agrees'].sum())} of {len(placed)} placed companies are most correlated with their own group.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
