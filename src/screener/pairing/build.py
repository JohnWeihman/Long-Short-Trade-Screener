"""Build the Stage 4 tables under data/pairs/ from the Stage 3 scores.

    python -m screener.pairing.build

Pairs are built for the same as-of date as the scores table; to change it,
re-run `python -m screener.scoring.build --as-of DATE` first. Reads only
local files. Run `python -m screener.pairing.deals` beforehand so that
acquisition targets are flagged and excluded.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from screener import config
from screener.cleaning.store import read_table, table_path, write_table
from screener.data.cache import Cache
from screener.data.prices import load_prices
from screener.pairing.candidates import candidate_pairs
from screener.pairing.config import PairingConfig, load_pairing_config
from screener.pairing.report import pairs_table
from screener.pairing.spread import SPREAD_COLUMNS, daily_returns, spread_series, spread_stats
from screener.pairing.stats import beta, betas, correlation, equal_weight_benchmark, hedge_ratio, share_ratio
from screener.scoring.peers import residual_returns, weekly_returns

AREA = "pairs"
SCORES_AREA = "scores"
DEAL_NOTES = {
    "pending_target": "BEING ACQUIRED (pending deal)",
    "pending_other": "party to a pending deal",
    "closed": "recently completed a deal; results may not be comparable",
}


def evaluate_pairs(
    candidates: pd.DataFrame, weekly: pd.DataFrame, daily: dict[str, pd.Series],
    price_now: dict[str, float], notes: dict[str, str], cfg: PairingConfig,
    deals: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pairs with statistics and pass/fail reasons, spread series of passing pairs).

    weekly: weekly returns, one column per company, already cut to the
    correlation window. daily: daily return series per ticker. deals maps
    ticker -> deal status (see pairing/deals.py); None means no deal check
    has been run.
    """
    deals = deals or {}
    benchmark = equal_weight_benchmark(weekly)
    beta_by = betas(weekly, benchmark, cfg.min_overlap_weeks)
    residual = residual_returns(weekly).corr(min_periods=cfg.min_overlap_weeks)
    empty = pd.Series(dtype="float64")

    rows, spreads = [], []
    for pair in candidates.itertuples(index=False):
        long, short = pair.long, pair.short
        have = long in weekly.columns and short in weekly.columns
        corr, weeks = correlation(weekly[long], weekly[short], cfg.min_overlap_weeks) if have else (float("nan"), 0)
        beta_long, beta_short = beta_by.get(long, float("nan")), beta_by.get(short, float("nan"))
        ratio = hedge_ratio(beta_long, beta_short)
        prices = price_now.get(long, float("nan")), price_now.get(short, float("nan"))
        spread = spread_series(pair.pair_id, daily.get(long, empty), daily.get(short, empty), ratio,
                               cfg.spread_history_days)

        passes_correlation = bool(not np.isnan(corr) and corr >= cfg.min_correlation)
        passes_hedge = bool(not np.isnan(ratio))
        reasons = []
        if not pair.passes_gap:
            reasons.append(f"score gap {pair.gap:.2f} below {cfg.min_composite_gap:.2f}")
        if not pair.passes_pillars:
            reasons.append(f"only {pair.pillars_favoring_long} pillar(s) favor the long")
        if not passes_correlation:
            reasons.append("too little shared price history" if np.isnan(corr)
                           else f"correlation {corr:.2f} below {cfg.min_correlation:.2f}")
        if not passes_hedge:
            reasons.append("no usable hedge ratio")
        status = {t: deals.get(t, "none") for t in (long, short)}
        targets = [t for t in (long, short) if status[t] == "pending_target"]
        passes_deal = not (targets and cfg.exclude_pending_targets)
        if not passes_deal:
            reasons.append(" and ".join(targets) + (" is" if len(targets) == 1 else " are")
                           + " being acquired (pending deal)")
        passes = not reasons

        pair_notes = [f"{t}: {DEAL_NOTES[status[t]]}" for t in (long, short) if status[t] in DEAL_NOTES]
        pair_notes += [f"{t}: {notes[t]}" for t in (long, short) if notes.get(t)]
        if pair.thin_group:
            pair_notes.insert(0, "thin peer group")
        rows.append({
            **pair._asdict(),
            "correlation": corr, "overlap_weeks": weeks,
            "residual_correlation": (residual.loc[long, short]
                                     if long in residual.index and short in residual.columns else float("nan")),
            "beta_long": beta_long, "beta_short": beta_short, "hedge_ratio": ratio,
            "pair_beta": beta(weekly[long], weekly[short], cfg.min_overlap_weeks) if have else float("nan"),
            "price_long": prices[0], "price_short": prices[1], "share_ratio": share_ratio(ratio, *prices),
            **spread_stats(spread, cfg.spread_zscore_window_days),
            "deal_long": status[long], "deal_short": status[short],
            "passes_correlation": passes_correlation, "passes_hedge": passes_hedge,
            "passes_deal": passes_deal, "passes": passes,
            "reasons": "; ".join(reasons), "notes": " | ".join(pair_notes),
        })
        if passes:
            spreads.append(spread)

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs, pd.DataFrame(columns=SPREAD_COLUMNS)
    order = pairs[pairs["passes"]].sort_values(["gap", "correlation"], ascending=False, kind="stable").index
    pairs["rank"] = pd.Series(range(1, len(order) + 1), index=order).reindex(pairs.index).astype("Int64")
    pairs = pairs.sort_values(["rank", "gap"], ascending=[True, False], na_position="last", kind="stable")
    all_spreads = pd.concat(spreads, ignore_index=True) if spreads else pd.DataFrame(columns=SPREAD_COLUMNS)
    return pairs.reset_index(drop=True), all_spreads


def build(cache: Cache) -> dict[str, pd.DataFrame]:
    cfg = load_pairing_config()
    scores = read_table(cache, "scores", SCORES_AREA)
    fundamentals = read_table(cache, "fundamentals", SCORES_AREA)
    as_of = pd.Timestamp(scores["as_of"].iloc[0])

    prices = {t: load_prices(cache, t) for t in scores["ticker"] if cache.prices_path("yfinance", t).exists()}
    weekly = weekly_returns(prices, cfg.correlation_window_weeks, as_of)
    daily = {t: daily_returns(p, as_of) for t, p in prices.items()}
    by_ticker = fundamentals.set_index("ticker")

    # Deal flags are optional input: without them every company counts as clear.
    deals = None
    if table_path(cache, "deals", AREA).exists():
        deals = read_table(cache, "deals", AREA).set_index("ticker")["status"].to_dict()

    pairs, spreads = evaluate_pairs(
        candidate_pairs(scores, cfg), weekly, daily,
        by_ticker["price"].to_dict(), by_ticker["notes"].fillna("").to_dict(), cfg, deals,
    )
    pairs["as_of"] = as_of
    pairs["deals_checked"] = deals is not None
    tables = {"pairs": pairs, "pair_spreads": spreads}
    for name, frame in tables.items():
        write_table(cache, name, frame, AREA)
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    cache = Cache(config.data_dir())
    try:
        tables = build(cache)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    pairs = tables["pairs"]
    as_of = pairs["as_of"].iloc[0] if len(pairs) else None
    print(f"Pairs as of {as_of:%Y-%m-%d}: {len(pairs)} candidates within peer groups")
    print(f"  {int(pairs['passes_gap'].sum()):>3} clear the score gap")
    print(f"  {int((pairs['passes_gap'] & pairs['passes_pillars']).sum()):>3} also have enough pillars favoring the long")
    score_ok = pairs["passes_gap"] & pairs["passes_pillars"]
    print(f"  {int((score_ok & pairs['passes_correlation'] & pairs['passes_hedge']).sum()):>3} "
          f"also clear the correlation filter and have a hedge ratio")
    if pairs["deals_checked"].iloc[0]:
        print(f"  {int(pairs['passes'].sum()):>3} also have no acquisition target on either side\n")
    else:
        print("\nWARNING: no deal check has been run, so acquisition targets are not excluded.\n"
              "         Run: python -m screener.pairing.deals   and then this command again.\n")
    print(pairs_table(pairs))
    print(f"\nSpread history saved for {tables['pair_spreads']['pair_id'].nunique()} pairs. "
          f"Written to {cache.root / AREA}")
    print("All candidates with reasons: python scripts/show_pairs.py --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
