"""Build the Stage 3 tables under data/scores/ from the cleaned tables.

    python -m screener.scoring.build                  # as of today
    python -m screener.scoring.build --as-of 2026-06-30

Reads only local files. Run `python -m screener.cleaning.build` first.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
import yaml

from screener import config
from screener.cleaning.store import read_table, write_table
from screener.cleaning.tagmap import load_tag_map
from screener.data.cache import Cache
from screener.data.prices import load_prices
from screener.scoring.fundamentals import Settings, build_fundamentals
from screener.scoring.metrics import compute_metrics
from screener.scoring.peers import load_peer_groups
from screener.scoring.report import group_table
from screener.scoring.scores import load_scoring_config, rank_metrics, score

DEFAULT_SCORING = config.PROJECT_ROOT / "config" / "scoring.yaml"
AREA = "scores"


def load_settings(path=DEFAULT_SCORING) -> Settings:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    known = {k: raw[k] for k in Settings.__dataclass_fields__ if k in raw}
    return Settings(**known)


def build(cache: Cache, as_of: pd.Timestamp, basis: str = "latest") -> dict[str, pd.DataFrame]:
    peers = load_peer_groups()
    prices = {
        t: load_prices(cache, t) for t in peers["ticker"] if cache.prices_path("yfinance", t).exists()
    }
    fundamentals = build_fundamentals(
        read_table(cache, "metrics_quarterly"), read_table(cache, "companies"), peers,
        load_tag_map(), prices, as_of, load_settings(), basis,
    )
    cfg = load_scoring_config()
    metrics = compute_metrics(fundamentals)
    metric_ranks = rank_metrics(metrics, cfg)
    tables = {
        "peer_groups": peers,
        "fundamentals": fundamentals,
        "metrics": metrics,
        "metric_ranks": metric_ranks,
        "scores": score(metric_ranks, metrics, cfg),
    }
    for name, frame in tables.items():
        write_table(cache, name, frame, AREA)
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--basis", choices=["latest", "first"], default="latest")
    args = parser.parse_args(argv)
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.today().normalize()

    cache = Cache(config.data_dir())
    try:
        tables = build(cache, as_of, args.basis)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    f = tables["fundamentals"]
    print(f"Fundamentals as of {as_of:%Y-%m-%d} ({args.basis} basis): {len(f)} companies\n")
    print(f"{'ticker':<6} {'group':<26} {'quarter':<9} {'ended':<11} {'mkt cap $B':>10} {'EV $B':>9} "
          f"{'rev TTM $B':>10} {'EBITDA $B':>9} {'FCF $B':>8} {'ROIC':>6}  notes")
    billions = lambda x: "-" if pd.isna(x) else f"{x / 1e9:,.1f}"  # noqa: E731
    for r in f.sort_values(["peer_group", "ticker"], na_position="last").itertuples():
        quarter = "-" if pd.isna(r.fiscal_year) else f"FY{int(r.fiscal_year)} Q{int(r.quarter)}"
        ended = "-" if pd.isna(r.period_end) else f"{r.period_end:%Y-%m-%d}"
        roic = "-" if pd.isna(r.roic) else f"{r.roic:.0%}"
        print(f"{r.ticker:<6} {str(r.peer_group or '(unplaced)'):<26} {quarter:<9} {ended:<11} "
              f"{billions(r.market_cap):>10} {billions(r.enterprise_value):>9} {billions(r.revenue_ttm):>10} "
              f"{billions(r.ebitda_ttm):>9} {billions(r.fcf_ttm):>8} {roic:>6}  {r.notes}")
    scores, metrics = tables["scores"], tables["metrics"]
    print("\nScores: rank within peer group, 0 = worst, 1 = best. dOM, accel and dFCFm are in percentage points.")
    for group in sorted(scores["peer_group"].dropna().unique()):
        print("\n" + group_table(scores, metrics, group))
    unplaced = scores[scores["peer_group"].isna()]["ticker"].tolist()
    if unplaced:
        print(f"\nNot ranked (no peer group): {', '.join(unplaced)}")
    print(f"\nWritten to {cache.root / AREA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
