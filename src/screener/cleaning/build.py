"""Rebuild the cleaned tables from the raw cache.

    python -m screener.cleaning.build

Reads only local files; never touches the network. Tables written so far
(under data/clean/): companies, filings, facts_raw, facts.
"""

from __future__ import annotations

import argparse

import pandas as pd

from screener import config
from screener.cleaning.companies import build_companies
from screener.cleaning.facts import dedupe, flag_scale_errors, flatten_company, mapped_tags
from screener.cleaning.metrics import resolve_all
from screener.cleaning.periods import assign_periods
from screener.cleaning.quarters import annual_table, quarterly_table
from screener.cleaning.splits import adjust_for_splits, load_splits
from screener.cleaning.store import write_table
from screener.cleaning.tagmap import load_tag_map
from screener.cleaning.validate import summary, validate
from screener.data.cache import Cache, read_gz_json
from screener.data.tickers import load_ticker_map, read_universe, split_known


def build(
    cache: Cache, ciks: list[str], tag_map: dict, price_tickers: dict[str, str] | None = None
) -> dict[str, pd.DataFrame]:
    """Build every cleaned table for the given CIKs and write them.

    price_tickers maps cik10 -> ticker of its cached price file, used for
    the split history. Without it no split adjustment is made.
    """
    companies, filings = build_companies(cache, ciks)
    splits = load_splits(cache, price_tickers or {})
    tags = mapped_tags(tag_map)
    raw_frames = [
        flatten_company(read_gz_json(cache.companyfacts_path(cik)), cik, tags)
        for cik in ciks
        if cache.companyfacts_path(cik).exists()
    ]
    facts_raw = adjust_for_splits(pd.concat(raw_frames, ignore_index=True), splits)
    facts_raw["scale_suspect"] = flag_scale_errors(facts_raw)
    facts = dedupe(facts_raw)
    metric_facts = resolve_all(facts, tag_map)
    cover_metrics = {name for name, metric in tag_map.items() if metric.kind == "cover"}
    assigned, years, quarters = assign_periods(metric_facts, cover_metrics)
    tables = {
        "companies": companies,
        "filings": filings,
        "splits": splits,
        "facts_raw": facts_raw,
        "facts": facts,
        "metric_facts": metric_facts,
        "fiscal_years": years,
        "fiscal_quarters": quarters,
        "metrics_quarterly": quarterly_table(assigned, quarters, tag_map),
        "metrics_annual": annual_table(assigned, quarters),
    }
    tables["validation"] = validate(tables, tag_map)
    for name, frame in tables.items():
        write_table(cache, name, frame)
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)

    cache = Cache(config.data_dir())
    known, missing = split_known(load_ticker_map(cache), read_universe(config.DEFAULT_UNIVERSE))
    price_tickers = dict(zip(known["cik10"], known["ticker"]))
    tables = build(cache, known["cik10"].tolist(), load_tag_map(), price_tickers)

    companies, facts = tables["companies"], tables["facts"]
    no_prices = [t for t in known["ticker"] if not cache.prices_path("yfinance", t).exists()]
    if no_prices:
        print(f"WARNING: no cached prices, so no split adjustment, for: {', '.join(no_prices)}")
    adjusted = tables["facts_raw"]["split_factor"] != 1
    print(f"splits    : {len(tables['splits']):>8,} rows  ({int(adjusted.sum()):,} share and per-share values "
          f"restated to today's share basis)")
    print(f"companies : {len(companies):>8,} rows  " + ", ".join(
        f"{n} {t}" for t, n in companies["filer_type"].value_counts().items()))
    print(f"filings   : {len(tables['filings']):>8,} rows")
    print(f"facts_raw : {len(tables['facts_raw']):>8,} rows  (every filing's version of every period)")
    print(f"facts     : {len(facts):>8,} rows  (one per period), {int(facts['restated'].sum()):,} restated")
    print(f"            {int(tables['facts_raw']['scale_suspect'].sum()):,} wrong-scale rows set aside "
          f"in {int((facts['scale_errors'] > 0).sum()):,} periods; "
          f"{int(facts['large_disagreement'].sum()):,} periods where filings disagree by 10x or more")
    mf, mq, ma = tables["metric_facts"], tables["metrics_quarterly"], tables["metrics_annual"]
    durations = int(mf["start"].notna().sum())
    print(f"metric_facts     : {len(mf):>8,} rows  (tag map applied to every reported period; "
          f"{int(mf['derived'].sum()):,} by formula)")
    print(f"fiscal_years     : {len(tables['fiscal_years']):>8,} rows   fiscal_quarters: {len(tables['fiscal_quarters']):,}")
    print(f"metrics_quarterly: {len(mq):>8,} rows  " + ", ".join(
        f"{n:,} {m}" for m, n in mq["method"].value_counts().items()))
    print(f"metrics_annual   : {len(ma):>8,} rows")
    print(f"            ({durations:,} reported durations in total; those that are not a fiscal quarter, "
          f"year-to-date or full year are left out)")
    counts = summary(tables["validation"])
    print(f"validation       : {len(tables['validation']):>8,} findings over each company's last 5 fiscal years")
    for row in counts.itertuples():
        print(f"    {row.severity:<8} {row.check:<28} {row.findings:>5}  in {row.companies} companies")
    print("    details: python -m screener.cleaning.validate")
    excluded = companies[~companies["include"]]
    for row in excluded.itertuples():
        print(f"  excluded: {row.ticker} ({row.name}) is '{row.filer_type}'")
    if missing:
        print(f"Not in SEC ticker file: {', '.join(missing)}")
    print(f"Written to {cache.root / 'clean'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
