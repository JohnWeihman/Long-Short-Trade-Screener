"""Print revenue, net income and shares outstanding for a company's most
recent fiscal year, straight from the cached raw EDGAR data, with the XBRL
tag, period and filing behind each number so it can be checked against the 10-K.

    python scripts/check_company.py NVDA

Reads the local cache only. Run `python -m screener.data.pull` first.
"""

from __future__ import annotations

import argparse
import sys

from screener import config
from screener.data.cache import Cache, read_gz_json
from screener.data.rawfacts import filing_url, latest_year_summary, period_days
from screener.data.tickers import UnknownTickerError, load_ticker_map, resolve_tickers

SECTIONS = [
    ("revenue", "REVENUE"),
    ("net_income", "NET INCOME"),
    ("shares_cover_page", "SHARES OUTSTANDING - 10-K cover page (dated after year end)"),
    ("shares_balance_sheet", "SHARES OUTSTANDING - balance sheet date"),
    ("shares_diluted_weighted", "DILUTED WEIGHTED-AVERAGE SHARES - full year (the EPS denominator)"),
]


def _value(row: dict) -> str:
    unit = row["unit"]
    return f"{row['val']:,.0f} {unit}" if isinstance(row["val"], (int, float)) else f"{row['val']} {unit}"


def _period(row: dict) -> str:
    if row.get("start"):
        return f"{row['start']} to {row['end']} ({period_days(row)} days)"
    return f"as of {row['end']}"


def _print_metric(title: str, metric: dict, cik: int) -> None:
    print(f"\n{title}")
    chosen = metric["chosen"]
    if chosen is None:
        print("  not found under the tags this script checks")
        return
    print(f"  value   : {_value(chosen)}")
    print(f"  tag     : {chosen['taxonomy']}:{chosen['tag']}")
    print(f"  period  : {_period(chosen)}")
    print(f"  filing  : {chosen['form']} filed {chosen['filed']}, accession {chosen['accn']}")
    print(f"  link    : {filing_url(cik, chosen['accn'])}")

    others = [r for r in metric["candidates"] if r is not chosen]
    if others:
        print("  other facts found for this period:")
        for r in others:
            flag = "" if r["val"] == chosen["val"] else "   <-- DIFFERENT VALUE"
            print(
                f"    {r['taxonomy']}:{r['tag']} = {_value(r)}  [{_period(r)}; "
                f"{r['form']} filed {r['filed']}, {r['accn']}]{flag}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ticker")
    args = parser.parse_args(argv)

    cache = Cache(config.data_dir())
    try:
        company = resolve_tickers(load_ticker_map(cache), [args.ticker]).iloc[0]
    except (FileNotFoundError, UnknownTickerError) as exc:
        print(exc, file=sys.stderr)
        return 1
    path = cache.companyfacts_path(company["cik10"])
    if not path.exists():
        print(
            f"No cached companyfacts for {company['ticker']} ({path}).\n"
            f"Run: python -m screener.data.pull --tickers {company['ticker']}",
            file=sys.stderr,
        )
        return 1

    summary = latest_year_summary(read_gz_json(path))
    print(f"{company['ticker']} - {summary['entity']} (CIK {company['cik10']})")
    print(f"Source file: {path}")
    if summary["fiscal_year_end"] is None:
        print("No full-year revenue or net income found in a 10-K under the tags this script checks.")
        return 1
    print(f"Most recent fiscal year ended: {summary['fiscal_year_end']}")
    for key, title in SECTIONS:
        _print_metric(title, summary[key], int(company["cik"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
