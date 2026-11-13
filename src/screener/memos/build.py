"""Build the Stage 5 tables under data/memos/.

    python -m screener.memos.build                  # fetch what is missing, extract, write memos
    python -m screener.memos.build --refresh        # re-download the filing documents
    python -m screener.memos.build --template-only  # no API calls this run

Every passing pair gets a template memo, rebuilt on every run so its numbers
are current. When ANTHROPIC_API_KEY is set, the top pairs also get a
model-written analyst view, regenerated only when a pair is new, a company
files, or a deal changes (see memos/model_writer.py). Every run appends a
row to data/memos/model_runs.parquet with its API calls, tokens and cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import pandas as pd

from screener import config
from screener.cleaning.store import read_table, table_path, write_table
from screener.data.cache import Cache, write_manifest
from screener.data.filings import latest_relevant_filing
from screener.data.http import SecClient
from screener.memos.facts import fact_sheet, filing_facts
from screener.memos.model_writer import ModelWriter, assemble, cache_key
from screener.memos.sources import (
    MemoConfig,
    extract_sections,
    fetch_documents,
    load_memo_config,
    select_documents,
    top_pair_tickers,
)
from screener.memos.write import company_sources, inputs_hash, render_template

AREA = "memos"
MEMO_COLUMNS = [
    "pair_id", "rank", "long", "short", "peer_group", "as_of", "memo", "memo_ai", "ai_status", "ai_note",
    "ai_model", "ai_written", "filings_through", "fact_sheet", "sources", "inputs_hash", "ai_key",
]
RUN_LOG_COLUMNS = [
    "run_at", "as_of", "ai_enabled", "ai_model", "note", "pairs", "ai_pairs", "cached", "generated",
    "fallbacks", "fallback_reasons", "api_calls", "input_tokens", "output_tokens", "estimated_cost_usd",
]


def _company_state(ticker: str, ciks: dict[str, str], filings: pd.DataFrame | None, as_of: pd.Timestamp,
                   deal_map: dict) -> tuple[str, str, str, pd.Timestamp | None]:
    """(ticker, latest relevant accession, deal status, its filing date)."""
    accn, filed = (None, None) if filings is None or ticker not in ciks else latest_relevant_filing(filings, ciks[ticker], as_of)
    return ticker, accn or "none", deal_map.get(ticker, {}).get("status", "none"), filed


def build_memos(
    pairs: pd.DataFrame, scores: pd.DataFrame, metric_ranks: pd.DataFrame, fundamentals: pd.DataFrame,
    deals: pd.DataFrame | None, sections: pd.DataFrame, documents: pd.DataFrame, cfg: MemoConfig,
    writer: ModelWriter | None = None, metrics: pd.DataFrame | None = None,
    quarterly: pd.DataFrame | None = None, filings: pd.DataFrame | None = None, ai_off_note: str = "",
) -> pd.DataFrame:
    """A template memo for every passing pair, plus a model-written memo for
    the top cfg.ai_pairs when a writer is given."""
    passing = pairs[pairs["passes"]].sort_values("rank")
    deal_map = {} if deals is None else deals.set_index("ticker")[["status", "evidence"]].to_dict("index")
    ciks = fundamentals.set_index("ticker")["cik10"].to_dict() if "cik10" in fundamentals.columns else {}
    rows = []
    for _, pair in passing.iterrows():
        rank = int(pair["rank"])
        as_of = pd.Timestamp(pair["as_of"])
        facts = fact_sheet(pair, scores, metric_ranks, fundamentals, deal_map, len(passing), metrics, quarterly)
        long_sources = company_sources(pair["long"], sections, documents, cfg.risk_headings)
        short_sources = company_sources(pair["short"], sections, documents, cfg.risk_headings)
        template = render_template(facts, long_sources, short_sources)
        states = [_company_state(t, ciks, filings, as_of, deal_map) for t in (pair["long"], pair["short"])]
        key = cache_key(pair["pair_id"], [s[:3] for s in states], cfg.model_writer)
        through = max((s[3] for s in states if s[3] is not None), default=None)
        row = {
            "pair_id": pair["pair_id"], "rank": rank, "long": pair["long"], "short": pair["short"],
            "peer_group": pair["peer_group"], "as_of": pair["as_of"], "memo": template, "memo_ai": None,
            "ai_status": "off", "ai_note": ai_off_note, "ai_model": None, "ai_written": None,
            "filings_through": None if through is None else f"{through:%Y-%m-%d}",
            "fact_sheet": json.dumps(facts), "sources": json.dumps([long_sources, short_sources]),
            "inputs_hash": inputs_hash(facts, long_sources, short_sources), "ai_key": key,
        }
        if writer is not None and rank <= cfg.ai_pairs:
            result = writer.write(filing_facts(facts), long_sources, short_sources, key)
            row.update(ai_status=result.status, ai_note=result.note)
            if result.body:
                row.update(ai_model=result.served_by, ai_written=result.written,
                           memo_ai=assemble(result.body, template, result.served_by, result.written,
                                            row["filings_through"] or "n/a"))
        elif writer is not None:
            row.update(ai_status="not top", ai_note=f"model-written only for the top {cfg.ai_pairs}")
        rows.append(row)
    return pd.DataFrame(rows, columns=MEMO_COLUMNS)


def run_log_row(memos: pd.DataFrame, writer: ModelWriter | None, cfg: MemoConfig, note: str) -> dict:
    """One line of the API call log for this run."""
    status = memos["ai_status"].value_counts().to_dict() if len(memos) else {}
    reasons = memos.loc[memos["ai_status"] == "fallback", "ai_note"].tolist() if len(memos) else []
    return {
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "as_of": memos["as_of"].iloc[0] if len(memos) else None,
        "ai_enabled": writer is not None, "ai_model": cfg.model_writer.model if writer else None, "note": note,
        "pairs": len(memos), "ai_pairs": min(cfg.ai_pairs, len(memos)) if writer else 0,
        "cached": status.get("cached", 0), "generated": status.get("generated", 0),
        "fallbacks": status.get("fallback", 0), "fallback_reasons": "; ".join(dict.fromkeys(reasons)),
        "api_calls": writer.calls if writer else 0,
        "input_tokens": writer.input_tokens if writer else 0, "output_tokens": writer.output_tokens if writer else 0,
        "estimated_cost_usd": round(writer.estimated_cost, 4) if writer else 0.0,
    }


def append_run_log(cache: Cache, row: dict) -> pd.DataFrame:
    """Add this run to data/memos/model_runs.parquet and return the whole log."""
    new = pd.DataFrame([row], columns=RUN_LOG_COLUMNS)
    if table_path(cache, "model_runs", AREA).exists():
        log = pd.concat([read_table(cache, "model_runs", AREA), new], ignore_index=True)
    else:
        log = new
    write_table(cache, "model_runs", log, AREA)
    return log


def write_markdown(cache: Cache, memos: pd.DataFrame) -> Path:
    """Each memo as a .md file (AI memos as *_ai.md), for reading outside the screener."""
    folder = cache.root / AREA / "md"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.md"):
        old.unlink()
    for memo in memos.itertuples():
        stem = f"{memo.rank:02d}_{memo.long}_{memo.short}"
        (folder / f"{stem}.md").write_text(memo.memo, encoding="utf-8")
        if isinstance(memo.memo_ai, str) and memo.memo_ai:
            (folder / f"{stem}_ai.md").write_text(memo.memo_ai, encoding="utf-8")
    return folder


def make_writer(cache: Cache, cfg: MemoConfig, enabled: bool = True) -> tuple[ModelWriter | None, str]:
    """(writer or None, one line saying whether AI memos will be written and why)."""
    if not enabled:
        return None, "AI memos off for this run (--template-only)."
    if not cfg.model_writer.enabled:
        return None, "AI memos off (model_writer.enabled is false in config/memos.yaml)."
    if not config.anthropic_api_key_present():
        return None, "AI memos off (no ANTHROPIC_API_KEY in .env)."
    return (ModelWriter(anthropic.Anthropic(), cfg.model_writer, cache.root / AREA / "model_cache"),
            f"AI memos on: {cfg.model_writer.model} (effort {cfg.model_writer.effort}) for the top "
            f"{cfg.ai_pairs} pairs, only where a pair is new or a company has filed.")


def build(cache: Cache, client, refresh: bool = False, writer: ModelWriter | None = None,
          ai_off_note: str = "") -> dict[str, pd.DataFrame]:
    cfg = load_memo_config()
    pairs = read_table(cache, "pairs", "pairs")
    filings = read_table(cache, "filings")
    companies = read_table(cache, "companies").set_index("ticker")["cik10"]
    as_of = pd.Timestamp(pairs["as_of"].iloc[0])

    ciks = {t: companies[t] for t in top_pair_tickers(pairs) if t in companies.index}
    documents, records = fetch_documents(client, cache, select_documents(filings, ciks, as_of, cfg.press_release_quarters), refresh)
    write_manifest(cache, records)
    sections = extract_sections(cache, documents, cfg)
    deals = read_table(cache, "deals", "pairs") if table_path(cache, "deals", "pairs").exists() else None
    memos = build_memos(
        pairs, read_table(cache, "scores", "scores"), read_table(cache, "metric_ranks", "scores"),
        read_table(cache, "fundamentals", "scores"), deals, sections, documents, cfg, writer,
        read_table(cache, "metrics", "scores"), read_table(cache, "metrics_quarterly"), filings, ai_off_note,
    )
    tables = {"documents": documents, "sections": sections, "memos": memos}
    for name, frame in tables.items():
        write_table(cache, name, frame, AREA)
    write_markdown(cache, memos)
    tables["model_runs"] = append_run_log(cache, run_log_row(memos, writer, cfg, ai_off_note))
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download filing documents")
    parser.add_argument("--template-only", action="store_true", help="no API calls this run")
    args = parser.parse_args(argv)
    try:
        client = SecClient(config.sec_user_agent())
    except config.ConfigError as exc:
        print(f"Configuration problem: {exc}", file=sys.stderr)
        return 2
    cache = Cache(config.data_dir())
    writer, writer_line = make_writer(cache, load_memo_config(), not args.template_only)
    print(writer_line)
    try:
        tables = build(cache, client, args.refresh, writer, "" if writer else writer_line)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    documents, sections = tables["documents"], tables["sections"]
    print(f"Documents: {len(documents)} for {documents['ticker'].nunique()} companies  ("
          + ", ".join(f"{n} {s}" for s, n in documents["status"].value_counts().items()) + ")")
    print(f"Sections : {len(sections)} extracted, {int((~sections['ok']).sum())} with a problem")
    for p in sections[~sections["ok"]].itertuples():
        print(f"  {p.ticker:<5} {p.kind:<17} {p.accn}  {p.section}: {p.problem}")

    memos = tables["memos"]
    print(f"\nMemos: {len(memos)} template memos (every passing pair)")
    for memo in memos.itertuples():
        ai = "" if memo.ai_status in ("off", "not top") else f"  AI: {memo.ai_status}" + (f" ({memo.ai_note})" if memo.ai_note else "")
        print(f"  {memo.rank:>2}. long {memo.long:<5} short {memo.short:<5} {len(memo.memo.split()):>4} words{ai}")
    run = tables["model_runs"].iloc[-1]
    print(f"\nAPI calls this run: {run['api_calls']}  (cached {run['cached']}, generated {run['generated']}, "
          f"fallbacks {run['fallbacks']}; {run['input_tokens']:,} input / {run['output_tokens']:,} output tokens, "
          f"about ${run['estimated_cost_usd']:.4f})")
    print(f"Run log: {table_path(cache, 'model_runs', AREA)} ({len(tables['model_runs'])} runs)")
    print(f"Written to {cache.root / AREA}  (Markdown files in {cache.root / AREA / 'md'})")
    print("Read one: python scripts/show_memo.py LONG SHORT   (add --ai for the AI version)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
