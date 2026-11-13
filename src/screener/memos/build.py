"""Build the Stage 5 tables under data/memos/.

    python -m screener.memos.build              # fetch what is missing, then extract
    python -m screener.memos.build --refresh    # re-download the filing documents

Fetches filing documents for the companies in the top pairs (cached; only
missing ones are downloaded), extracts the sections memos are built from,
and writes one memo per top pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from screener import config
from screener.cleaning.store import read_table, table_path, write_table
from screener.data.cache import Cache, write_manifest
from screener.data.http import SecClient
from screener.memos.facts import fact_sheet
from screener.memos.write import company_sources, inputs_hash, render_template
from screener.memos.sources import (
    MemoConfig,
    extract_sections,
    fetch_documents,
    load_memo_config,
    select_documents,
    top_pair_tickers,
)

AREA = "memos"
MEMO_COLUMNS = ["pair_id", "rank", "long", "short", "peer_group", "as_of", "writer", "memo",
                "fact_sheet", "sources", "inputs_hash"]


def build_memos(
    pairs: pd.DataFrame, scores: pd.DataFrame, metric_ranks: pd.DataFrame, fundamentals: pd.DataFrame,
    deals: pd.DataFrame | None, sections: pd.DataFrame, documents: pd.DataFrame, cfg: MemoConfig,
) -> pd.DataFrame:
    """One template memo per top-ranked passing pair."""
    passing = pairs[pairs["passes"]].sort_values("rank")
    deal_map = {} if deals is None else deals.set_index("ticker")[["status", "evidence"]].to_dict("index")
    rows = []
    for _, pair in passing.head(cfg.top_pairs).iterrows():
        facts = fact_sheet(pair, scores, metric_ranks, fundamentals, deal_map, len(passing))
        long_sources = company_sources(pair["long"], sections, documents, cfg.risk_headings)
        short_sources = company_sources(pair["short"], sections, documents, cfg.risk_headings)
        rows.append({
            "pair_id": pair["pair_id"], "rank": int(pair["rank"]), "long": pair["long"], "short": pair["short"],
            "peer_group": pair["peer_group"], "as_of": pair["as_of"], "writer": "template",
            "memo": render_template(facts, long_sources, short_sources),
            "fact_sheet": json.dumps(facts), "sources": json.dumps([long_sources, short_sources]),
            "inputs_hash": inputs_hash(facts, long_sources, short_sources),
        })
    return pd.DataFrame(rows, columns=MEMO_COLUMNS)


def write_markdown(cache: Cache, memos: pd.DataFrame) -> Path:
    """Each memo as a .md file, for reading outside the screener."""
    folder = cache.root / AREA / "md"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.md"):
        old.unlink()
    for memo in memos.itertuples():
        (folder / f"{memo.rank:02d}_{memo.long}_{memo.short}.md").write_text(memo.memo, encoding="utf-8")
    return folder


def build(cache: Cache, client, refresh: bool = False) -> dict[str, pd.DataFrame]:
    cfg = load_memo_config()
    pairs = read_table(cache, "pairs", "pairs")
    filings = read_table(cache, "filings")
    companies = read_table(cache, "companies").set_index("ticker")["cik10"]
    as_of = pd.Timestamp(pairs["as_of"].iloc[0])

    tickers = top_pair_tickers(pairs, cfg.top_pairs)
    ciks = {t: companies[t] for t in tickers if t in companies.index}
    documents, records = fetch_documents(client, cache, select_documents(filings, ciks, as_of, cfg.press_release_quarters), refresh)
    write_manifest(cache, records)
    sections = extract_sections(cache, documents, cfg)
    deals = read_table(cache, "deals", "pairs") if table_path(cache, "deals", "pairs").exists() else None
    memos = build_memos(
        pairs, read_table(cache, "scores", "scores"), read_table(cache, "metric_ranks", "scores"),
        read_table(cache, "fundamentals", "scores"), deals, sections, documents, cfg,
    )
    tables = {"documents": documents, "sections": sections, "memos": memos}
    for name, frame in tables.items():
        write_table(cache, name, frame, AREA)
    write_markdown(cache, memos)
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download filing documents")
    args = parser.parse_args(argv)
    try:
        client = SecClient(config.sec_user_agent())
    except config.ConfigError as exc:
        print(f"Configuration problem: {exc}", file=sys.stderr)
        return 2
    cache = Cache(config.data_dir())
    try:
        tables = build(cache, client, args.refresh)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    documents, sections = tables["documents"], tables["sections"]
    print(f"Documents: {len(documents)} for {documents['ticker'].nunique()} companies  ("
          + ", ".join(f"{n} {s}" for s, n in documents["status"].value_counts().items()) + ")")
    print(f"Sections : {len(sections)} extracted, {int((~sections['ok']).sum())} with a problem\n")
    wide = sections.assign(cell=sections["chars"].map("{:,}".format).where(sections["ok"], "FAIL"))
    annual = wide[wide["kind"] == "annual_report"].pivot_table(
        index="ticker", columns="section", values="cell", aggfunc="first", sort=False)
    print("10-K sections (characters):")
    print(annual.to_string())
    release = sections[sections["kind"] == "earnings_release"]
    ok = release.groupby("ticker")["ok"].agg(["sum", "size"])
    print("\nEarnings releases: sections extracted / expected")
    print("  " + "  ".join(f"{t} {int(r['sum'])}/{int(r['size'])}" for t, r in ok.iterrows()))
    problems = sections[~sections["ok"]]
    if len(problems):
        print("\nProblems:")
        for p in problems.itertuples():
            print(f"  {p.ticker:<5} {p.kind:<17} {p.accn}  {p.section}: {p.problem}")
    memos = tables["memos"]
    print(f"\nMemos: {len(memos)} written (template) for pairs ranked 1 to {len(memos)}")
    for memo in memos.itertuples():
        print(f"  {memo.rank:>2}. long {memo.long:<5} short {memo.short:<5} {len(memo.memo.split()):>4} words")
    print(f"\nWritten to {cache.root / AREA}  (Markdown files in {cache.root / AREA / 'md'})")
    print("Read one: python scripts/show_memo.py LONG SHORT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
