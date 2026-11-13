"""Which filings a memo draws on, fetching them, and extracting their sections.

Two tables come out of this:
documents   one row per filing document used (10-K, earnings press release)
sections    one row per extracted section, with `ok` / `problem`
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd
import yaml

from screener import config
from screener.data.cache import BytesFetcher, Cache, FetchRecord
from screener.data.filings import (
    document_url,
    earnings_releases,
    fetch_document,
    latest_annual_report,
    parse_submission,
    press_release,
    read_document,
    submission_file_name,
)
from screener.data.rawfacts import filing_url
from screener.memos.extract import emphasized_lines, html_to_lines, risk_headings, split_10k, split_press_release

DEFAULT_MEMOS = config.PROJECT_ROOT / "config" / "memos.yaml"
DOCUMENT_COLUMNS = ["cik10", "ticker", "kind", "accn", "form", "filed", "report_date", "file_name", "url", "status"]
SECTION_COLUMNS = ["cik10", "ticker", "kind", "accn", "filed", "section", "text", "chars", "ok", "problem"]
ANNUAL, EARNINGS = "annual_report", "earnings_release"


@dataclass(frozen=True)
class MemoConfig:
    top_pairs: int = 10
    press_release_quarters: int = 4
    risk_headings: int = 5


def load_memo_config(path: Path = DEFAULT_MEMOS) -> MemoConfig:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    unknown = set(raw) - {f.name for f in fields(MemoConfig)}
    if unknown:
        raise ValueError(f"unknown memo settings: {sorted(unknown)}")
    return MemoConfig(**raw)


def top_pair_tickers(pairs: pd.DataFrame, count: int) -> list[str]:
    """Tickers in the top `count` passing pairs, in rank order, each once."""
    top = pairs[pairs["passes"]].sort_values("rank").head(count)
    return list(dict.fromkeys(top[["long", "short"]].to_numpy().ravel().tolist()))


def select_documents(filings: pd.DataFrame, ciks: dict[str, str], as_of: pd.Timestamp, quarters: int) -> pd.DataFrame:
    """The filings to use for each ticker: latest 10-K and recent earnings 8-Ks.

    ciks maps ticker -> cik10. A 10-K is read from its main document; an
    8-K from the complete submission, which contains the press release.
    """
    rows = []
    for ticker, cik10 in ciks.items():
        annual = latest_annual_report(filings, cik10, as_of)
        chosen = ([] if annual is None else [(ANNUAL, annual, annual["primary_document"])]) + [
            (EARNINGS, row, submission_file_name(row["accn"]))
            for _, row in earnings_releases(filings, cik10, as_of, quarters).iterrows()
        ]
        for kind, row, file_name in chosen:
            rows.append({
                "cik10": cik10, "ticker": ticker, "kind": kind, "accn": row["accn"], "form": row["form"],
                "filed": row["filed"], "report_date": row["report_date"], "file_name": file_name,
                "url": filing_url(cik10, row["accn"]), "status": "selected",
            })
    return pd.DataFrame(rows, columns=DOCUMENT_COLUMNS)


def fetch_documents(
    client: BytesFetcher, cache: Cache, documents: pd.DataFrame, refresh: bool = False
) -> tuple[pd.DataFrame, list[FetchRecord]]:
    """Download whatever is not cached; status becomes downloaded / cached / not_found / error."""
    records, statuses = [], []
    for doc in documents.itertuples():
        try:
            record = fetch_document(client, cache, doc.cik10, doc.accn, doc.file_name, refresh)
        except Exception as exc:  # one bad filing must not stop the rest
            record = FetchRecord("edgar/filings", f"{doc.cik10}/{doc.accn}/{doc.file_name}",
                                 document_url(doc.cik10, doc.accn, doc.file_name), "", "error", 0, "",
                                 f"{type(exc).__name__}: {exc}")
        records.append(record)
        statuses.append(record.status)
    return documents.assign(status=statuses), records


def _rows(doc, sections, extra=()) -> list[dict]:
    base = {"cik10": doc.cik10, "ticker": doc.ticker, "kind": doc.kind, "accn": doc.accn, "filed": doc.filed}
    out = [{**base, "section": s.name, "text": s.text, "chars": s.chars, "ok": s.ok, "problem": s.problem}
           for s in sections]
    return out + [{**base, **row} for row in extra]


def extract_sections(cache: Cache, documents: pd.DataFrame, cfg: MemoConfig = MemoConfig()) -> pd.DataFrame:
    """Sections of every cached document. Documents that were not fetched,
    or cannot be read, produce one failed row saying why."""
    rows: list[dict] = []
    for doc in documents.itertuples():
        failed = lambda why: _rows(doc, [], [{"section": "document", "text": "", "chars": 0, "ok": False, "problem": why}])  # noqa: E731
        if not cache.filing_path(doc.cik10, doc.accn, doc.file_name).exists():
            rows += failed(f"not in the cache ({doc.status})")
            continue
        raw = read_document(cache, doc.cik10, doc.accn, doc.file_name)
        if doc.kind == ANNUAL:
            sections = split_10k(html_to_lines(raw))
            headings = risk_headings(sections["risk_factors"].text, emphasized_lines(raw), cfg.risk_headings)
            text = "\n".join(headings)
            rows += _rows(doc, sections.values(), [{
                "section": "risk_headings", "text": text, "chars": len(text), "ok": bool(headings),
                "problem": "" if headings else "no bold headings found in the risk section",
            }])
        else:
            exhibit = press_release(parse_submission(raw))
            if exhibit is None:
                rows += failed("no press release exhibit (EX-99) in the 8-K")
                continue
            rows += _rows(doc, split_press_release(html_to_lines(exhibit["text"])).values())
    return pd.DataFrame(rows, columns=SECTION_COLUMNS)
