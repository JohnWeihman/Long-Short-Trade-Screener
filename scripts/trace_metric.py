"""Show a cleaned metric next to the raw EDGAR facts behind it.

    python scripts/trace_metric.py NVDA revenue                 # recent quarters, trace the latest
    python scripts/trace_metric.py NVDA revenue --fy 2026 --quarter 4
    python scripts/trace_metric.py KLAC operating_income --annual --fy 2026

Reads the cleaned tables only. Run `python -m screener.cleaning.build` first.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from screener import config
from screener.cleaning.store import read_table
from screener.cleaning.tagmap import load_tag_map
from screener.cleaning.trace import periods_behind, raw_behind
from screener.data.cache import Cache
from screener.data.rawfacts import filing_url

TABLES = ["companies", "metric_facts", "metrics_quarterly", "metrics_annual",
          "fiscal_years", "fiscal_quarters", "facts_raw", "validation"]


def number(x) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{x:,.4f}" if abs(x) < 1000 else f"{x:,.0f}"


def day(x) -> str:
    return "-" if x is None or pd.isna(x) else f"{x:%Y-%m-%d}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ticker")
    parser.add_argument("metric")
    parser.add_argument("--fy", type=int, help="fiscal year to trace (default: the latest)")
    parser.add_argument("--quarter", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--annual", action="store_true", help="show and trace fiscal years instead of quarters")
    parser.add_argument("--rows", type=int, default=8, help="how many recent periods to list")
    args = parser.parse_args(argv)

    cache = Cache(config.data_dir())
    tag_map = load_tag_map()
    if args.metric not in tag_map:
        print(f"Unknown metric '{args.metric}'. Choose from: {', '.join(tag_map)}", file=sys.stderr)
        return 1
    try:
        tables = {name: read_table(cache, name) for name in TABLES}
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    company = tables["companies"][tables["companies"]["ticker"] == args.ticker.upper()]
    if company.empty:
        print(f"{args.ticker.upper()} is not in the cleaned company table.", file=sys.stderr)
        return 1
    cik10, cik = company["cik10"].iloc[0], int(company["cik10"].iloc[0])
    unit = tag_map[args.metric].unit

    source = tables["metrics_annual"] if args.annual else tables["metrics_quarterly"]
    series = source[(source["cik10"] == cik10) & (source["metric"] == args.metric)]
    if series.empty:
        print(f"No cleaned values for {args.ticker.upper()} {args.metric}.")
        return 1

    print(f"{args.ticker.upper()} - {company['name'].iloc[0]} - {args.metric} ({unit})")
    print("Share and per-share figures are on today's share basis.\n")
    header = f"{'period':<10} {'ended':<11} {'value':>20} {'first reported':>20}  {'how':<9} {'known on':<11} source"
    print(header)
    for row in series.tail(args.rows).itertuples():
        label = f"FY{row.fiscal_year}" + ("" if args.annual else f" Q{row.quarter}")
        how = "reported" if args.annual else row.method
        flag = " (formula)" if row.formula_derived else ""
        print(f"{label:<10} {day(row.period_end):<11} {number(row.val):>20} {number(row.val_first):>20}  "
              f"{how:<9} {day(row.known_on):<11} {row.source}{flag}")

    last = series.iloc[-1]
    fiscal_year = args.fy or int(last["fiscal_year"])
    quarter = None if args.annual else (args.quarter or (int(last["quarter"]) if not args.fy else 4))
    label = f"FY{fiscal_year}" + ("" if quarter is None else f" Q{quarter}")

    target = series[series["fiscal_year"] == fiscal_year]
    if quarter is not None:
        target = target[target["quarter"] == quarter]
    if target.empty:
        print(f"\nNo value for {label}.")
        return 1
    row = target.iloc[0]
    print(f"\n=== {label}: {number(row['val'])} ===")
    if quarter is not None and row["method"] == "ratio":
        top, bottom = tag_map[args.metric].ratio_of
        print(f"Computed as {top} / {bottom} for the quarter. Trace those two metrics to see the facts behind it.")
        return 0
    if quarter is not None and row["method"] == "derived":
        print("Derived: year-to-date through this quarter minus year-to-date through the previous quarter.")

    periods = periods_behind(tables, tag_map, cik10, args.metric, fiscal_year, quarter)
    print("\nReported periods used (after tag selection):")
    for p in periods.itertuples():
        span = f"as of {day(p.end)}" if pd.isna(p.start) else f"{day(p.start)} to {day(p.end)} ({(p.end - p.start).days + 1} days)"
        print(f"  {span:<42} {number(p.val):>20}   first reported {number(p.val_first):>20}   via {p.source}")

    raw = raw_behind(tables, tag_map, cik10, args.metric, periods)
    print("\nRaw EDGAR facts for those periods (every filing's version):")
    for r in raw.itertuples():
        span = f"as of {day(r.end)}" if pd.isna(r.start) else f"{day(r.start)} to {day(r.end)}"
        notes = []
        if r.split_factor != 1:
            notes.append(f"filed as {number(r.val_reported)}, split factor {r.split_factor:g}")
        if r.scale_suspect:
            notes.append("WRONG SCALE - set aside")
        if isinstance(r.form, str) and "14A" in r.form:
            notes.append("proxy - not used")
        print(f"  {r.tag:<52} {span:<26} {number(r.val):>20}  {r.form:<7} filed {day(r.filed)}  {r.accn}"
              + (f"   [{'; '.join(notes)}]" if notes else ""))
    if len(raw):
        latest = raw.sort_values("filed").iloc[-1]
        print(f"\nMost recent filing above: {filing_url(cik, latest['accn'])}")

    findings = tables["validation"]
    findings = findings[(findings["cik10"] == cik10) & (findings["fiscal_year"] == fiscal_year)
                        & (findings["metric"] == args.metric)]
    if quarter is not None:
        findings = findings[findings["quarter"].isna() | (findings["quarter"] == quarter)]
    if len(findings):
        print("\nValidation findings for this value:")
        for f in findings.itertuples():
            print(f"  [{f.severity}] {f.check}: {f.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
