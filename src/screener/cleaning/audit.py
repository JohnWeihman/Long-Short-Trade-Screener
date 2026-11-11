"""Tag-map coverage audit: for each company's recent fiscal years, which
metrics the tag map finds, under which tag, and which it misses.

    python -m screener.cleaning.audit            # last 5 fiscal years
    python -m screener.cleaning.audit --years 10

Annual periods only. It answers "does the tag map reach this company's
numbers at all"; quarterly logic is built on top once the map is right.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd

from screener import config
from screener.cleaning.tagmap import Formula, Metric, Term, load_tag_map
from screener.data.cache import Cache, read_gz_json
from screener.data.rawfacts import ANNUAL_FORMS, is_annual_duration, tag_rows
from screener.data.tickers import load_ticker_map, read_universe, split_known

COVER_WINDOW_DAYS = 150
COLUMNS = [
    "metric", "kind", "optional", "fy_end", "found", "derived", "taxonomy", "tag",
    "formula", "val", "accn", "filed", "tags_with_data",
]


def fiscal_year_ends(companyfacts: dict, tag_map: dict[str, Metric], years: int) -> list[str]:
    """The most recent fiscal year-end dates, newest first, taken from
    full-year revenue and net income facts in annual filings."""
    ends = {
        row["end"]
        for name in ("revenue", "net_income")
        for taxonomy, tag in tag_map[name].tags
        for row in tag_rows(companyfacts, taxonomy, tag)
        if is_annual_duration(row)
    }
    return sorted(ends, reverse=True)[:years]


def _matches(metric: Metric, row: dict, fy_end: str) -> bool:
    if row["unit"] != metric.unit or row.get("form") not in ANNUAL_FORMS:
        return False
    if metric.kind == "duration":
        return is_annual_duration(row) and row["end"] == fy_end
    if row.get("start"):
        return False
    if metric.kind == "instant":
        return row["end"] == fy_end
    latest = (date.fromisoformat(fy_end) + timedelta(days=COVER_WINDOW_DAYS)).isoformat()
    return fy_end < row["end"] <= latest  # cover page: dated after the period end


def _latest_row(companyfacts: dict, metric: Metric, taxonomy: str, tag: str, fy_end: str) -> dict | None:
    """The most recently filed value of one tag for the metric's period."""
    rows = [r for r in tag_rows(companyfacts, taxonomy, tag) if _matches(metric, r, fy_end)]
    return max(rows, key=lambda r: (r.get("filed", ""), r.get("accn", ""))) if rows else None


def _term_value(companyfacts: dict, tag_map: dict[str, Metric], metric: Metric, term: Term, fy_end: str) -> float | None:
    if term.is_metric:
        return find_metric(companyfacts, tag_map, term.name, fy_end)["val"]
    row = _latest_row(companyfacts, metric, term.taxonomy, term.name, fy_end)
    return float(row["val"]) if row else None


def apply_formula(
    companyfacts: dict, tag_map: dict[str, Metric], metric: Metric, formula: Formula, fy_end: str
) -> tuple[float, str] | None:
    """(value, 'A + B - C' text of the terms actually used), or None if a
    required term is missing or no term has data."""
    total = 0.0
    text = ""
    for terms, sign, required in (
        (formula.add, 1, True),
        (formula.subtract, -1, True),
        (formula.add_if_present, 1, False),
        (formula.subtract_if_present, -1, False),
    ):
        for term in terms:
            value = _term_value(companyfacts, tag_map, metric, term, fy_end)
            if value is None:
                if required:
                    return None
                continue
            total += sign * value
            text += (" + " if sign > 0 else " - ") + term.label
    if not text:
        return None
    return total, text.removeprefix(" + ").strip()


def find_metric(companyfacts: dict, tag_map: dict[str, Metric], name: str, fy_end: str) -> dict:
    """The value the tag map selects for one metric and fiscal year.

    First tag in priority order with data (its most recently filed value);
    if no tag has data, the first formula that can be computed.
    """
    metric = tag_map[name]
    chosen = None
    tags_with_data = []
    for taxonomy, tag in metric.tags:
        row = _latest_row(companyfacts, metric, taxonomy, tag, fy_end)
        if row is None:
            continue
        tags_with_data.append(tag)
        chosen = chosen or row

    derived = None
    if chosen is None:
        for formula in metric.formulas:
            derived = apply_formula(companyfacts, tag_map, metric, formula, fy_end)
            if derived is not None:
                break

    return {
        "metric": metric.name,
        "kind": metric.kind,
        "optional": metric.optional,
        "fy_end": fy_end,
        "found": chosen is not None or derived is not None,
        "derived": derived is not None,
        "taxonomy": chosen["taxonomy"] if chosen else None,
        "tag": chosen["tag"] if chosen else None,
        "formula": derived[1] if derived else None,
        "val": float(chosen["val"]) if chosen else (derived[0] if derived else None),
        "accn": chosen["accn"] if chosen else None,
        "filed": chosen["filed"] if chosen else None,
        "tags_with_data": ", ".join(tags_with_data),
    }


def company_coverage(companyfacts: dict, tag_map: dict[str, Metric], years: int = 5) -> pd.DataFrame:
    """One row per (metric, fiscal year) for one company."""
    rows = [
        find_metric(companyfacts, tag_map, name, fy_end)
        for fy_end in fiscal_year_ends(companyfacts, tag_map, years)
        for name in tag_map
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def coverage_matrix(coverage: pd.DataFrame) -> pd.DataFrame:
    """Company x metric: number of fiscal years found."""
    counts = coverage.groupby(["ticker", "metric"], sort=False)["found"].sum().unstack("metric")
    return counts.astype(int)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, default=5)
    args = parser.parse_args(argv)

    cache = Cache(config.data_dir())
    tag_map = load_tag_map()
    companies, missing = split_known(load_ticker_map(cache), read_universe(config.DEFAULT_UNIVERSE))
    frames = []
    for row in companies.itertuples(index=False):
        frame = company_coverage(read_gz_json(cache.companyfacts_path(row.cik10)), tag_map, args.years)
        frame.insert(0, "ticker", row.ticker)
        frame.insert(1, "cik10", row.cik10)
        frames.append(frame)
    coverage = pd.concat(frames, ignore_index=True)

    out = cache.root / "clean" / "audit" / "tag_coverage.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_parquet(out, index=False)

    matrix = coverage_matrix(coverage)
    years_per_company = coverage.groupby("ticker", sort=False)["fy_end"].nunique()
    print(f"Fiscal years found per metric (out of up to {args.years}); '.' = none\n")
    names = list(tag_map)
    for i, name in enumerate(names, start=1):
        flag = " (optional)" if tag_map[name].optional else ""
        print(f"  {i:>2} {name}{flag}")
    print("\n" + "tick  yrs " + " ".join(f"{i:>2}" for i in range(1, len(names) + 1)))
    for ticker in matrix.index:
        cells = " ".join(f"{(str(matrix.loc[ticker, n]) if matrix.loc[ticker, n] else '.'):>2}" for n in names)
        print(f"{ticker:<5} {years_per_company[ticker]:>3} {cells}")

    required = coverage[~coverage["optional"]]
    gaps = required[~required["found"]]
    print(f"\nRequired metrics: {len(required) - len(gaps)} of {len(required)} company-years found "
          f"({1 - len(gaps) / len(required):.1%}), {int(required['derived'].sum())} of them by formula")

    derived = coverage[coverage["derived"]]
    if len(derived):
        print("\nFilled by formula (company-years):")
        for (metric, formula), group in derived.groupby(["metric", "formula"], sort=False):
            tickers = group.groupby("ticker", sort=False).size()
            print(f"  {metric} = {formula}: " + ", ".join(f"{t} x{n}" for t, n in tickers.items()))
    if len(gaps):
        print("\nRequired metrics still missing:")
        for (ticker, metric), group in gaps.groupby(["ticker", "metric"], sort=False):
            print(f"  {ticker:<5} {metric}: FY ending " + ", ".join(sorted(group["fy_end"])))
    if missing:
        print(f"Not in SEC ticker file: {', '.join(missing)}")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
