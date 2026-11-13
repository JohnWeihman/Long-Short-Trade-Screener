"""Detect companies involved in mergers and acquisitions.

A stock under a signed acquisition trades on the deal terms, not on its
fundamentals, so a low score is no reason to short it. This module reads
each company's latest earnings press release for deal language and
classifies the company:

    pending_target   it is being acquired            -> its pairs are excluded
    pending_other    party to a pending deal          -> flagged only
    closed           recently completed a deal        -> flagged only
    none

Keyword detection is imperfect: it misses deals announced between earnings
releases and can misread routine language. config/deals.csv overrides it.

    python -m screener.pairing.deals            # fetch missing releases, detect, write the table
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from screener import config
from screener.cleaning.store import read_table, write_table
from screener.data.cache import BytesFetcher, Cache, write_manifest
from screener.data.filings import (
    earnings_releases,
    fetch_document,
    parse_submission,
    press_release,
    read_document,
    submission_file_name,
)
from screener.data.http import SecClient
from screener.data.rawfacts import filing_url
from screener.memos.extract import html_to_lines

DEFAULT_OVERRIDES = config.PROJECT_ROOT / "config" / "deals.csv"
AREA = "pairs"
STATUSES = ("pending_target", "pending_other", "closed", "none")
COLUMNS = ["cik10", "ticker", "status", "source", "evidence", "accn", "filed", "url"]
EVIDENCE_CHARS = 320

_PENDING = re.compile(
    r"pending (?:acquisition|merger|transaction|combination)"
    r"|proposed (?:acquisition|merger|transaction|combination)"
    r"|definitive (?:merger |transaction )?agreement"
    r"|agree[sd]? to (?:acquire|be acquired|merge)"
    r"|merger agreement|to be acquired by", re.I)
# A pending-deal phrase only counts in a sentence that is about M&A
# ("subject to definitive agreements" for a partnership is not a deal).
_DEAL_WORD = re.compile(r"\b(?:acquir\w+|acquisitions?|merg\w+|combination)\b", re.I)
# "completed its acquisition of ElevATE" counts; "charges related to completed
# acquisitions of $0.01 per share" does not: a capitalized name must follow.
_CLOSED = re.compile(
    r"(?i:(?:completed|completion of|closed|closing of|consummat\w+) (?:the |its |our )?(?:previously announced )?"
    r"(?:acquisitions?|merger|combination) (?:of|with)) [A-Z]"
    r"|(?i:following (?:the |its |our )?[\w ]{0,40}?(?:combination|merger))"
    r"|(?i:as a combined company)")
# Legal boilerplate lists every conceivable risk, including deals that do not exist.
_BOILERPLATE = re.compile(r"forward-looking statements|safe harbor", re.I)
# Signs that THIS company is the one being bought.
_TARGET_GENERIC = re.compile(
    r"suspend\w* (?:providing |issuing |its |our )?[\w -]{0,30}?guidance"
    r"|(?:will not|does not plan to|is not|not be) (?:be )?(?:host|hold|conduct)\w* [\w ]{0,40}?(?:conference call|earnings call|webcast)"
    r"|to be acquired by", re.I)
_TARGET_NAMED = (
    r"acquisition of {name}",
    r"acquire (?:all (?:of )?(?:the )?(?:outstanding )?(?:shares of )?)?{name}",
    r"merger of {name} (?:with|into)",
)


def _name_pattern(company_name: str | None) -> re.Pattern | None:
    """Matches deal phrases naming this company, using the first word of its
    name ('SILICON LABORATORIES INC' -> 'silicon', which also matches
    'Silicon Labs'). Skipped for very short first words, which match too much."""
    words = re.findall(r"[A-Za-z]+", company_name or "")
    if not words or len(words[0]) < 4:
        return None
    name = re.escape(words[0])
    return re.compile("|".join(p.format(name=name) for p in _TARGET_NAMED), re.I)


def detect_deal(lines: list[str], company_name: str | None = None) -> tuple[str, str]:
    """(status, the sentence that triggered it) for one press release."""
    named = _name_pattern(company_name)
    lines = [line for line in lines if not _BOILERPLATE.search(line)]
    pending = [line for line in lines if _PENDING.search(line) and _DEAL_WORD.search(line)]
    closed = [line for line in lines if _CLOSED.search(line)]
    target = [line for line in lines if _TARGET_GENERIC.search(line) or (named and named.search(line))]

    def evidence(candidates: list[str]) -> str:
        return candidates[0][:EVIDENCE_CHARS]

    if pending and target:
        both = [line for line in target if _PENDING.search(line)]
        return "pending_target", evidence(both or target)
    if pending:
        return "pending_other", evidence(pending)
    if closed:
        return "closed", evidence(closed)
    return "none", ""


def load_overrides(path: Path = DEFAULT_OVERRIDES) -> pd.DataFrame:
    """Manual flags: ticker, status, note."""
    frame = pd.read_csv(path, dtype=str, comment="#", keep_default_na=False)
    if list(frame.columns) != ["ticker", "status", "note"]:
        raise ValueError(f"{path} must have exactly the columns ticker,status,note")
    frame["ticker"] = frame["ticker"].str.strip().str.upper()
    frame["status"] = frame["status"].str.strip()
    bad = sorted(set(frame["status"]) - set(STATUSES))
    if bad:
        raise ValueError(f"{path}: unknown status {bad}; use one of {STATUSES}")
    return frame.drop_duplicates("ticker", keep="last").reset_index(drop=True)


def detect_all(
    client: BytesFetcher | None, cache: Cache, filings: pd.DataFrame, companies: pd.DataFrame,
    as_of: pd.Timestamp, overrides: pd.DataFrame, refresh: bool = False,
) -> tuple[pd.DataFrame, list]:
    """One row per company: detected status from its latest earnings release,
    replaced by the manual override if there is one.

    With client=None nothing is downloaded: companies whose release is not
    cached are reported as status 'none' with the reason in `evidence`.
    """
    manual = overrides.set_index("ticker")
    rows, records = [], []
    for company in companies.itertuples():
        row = {"cik10": company.cik10, "ticker": company.ticker, "status": "none", "source": "detected",
               "evidence": "", "accn": None, "filed": pd.NaT, "url": None}
        latest = earnings_releases(filings, company.cik10, as_of, 1)
        if latest.empty:
            row["evidence"] = "no earnings release found to check"
        else:
            release = latest.iloc[0]
            name = submission_file_name(release["accn"])
            row.update(accn=release["accn"], filed=release["filed"], url=filing_url(company.cik10, release["accn"]))
            if client is not None:
                try:
                    records.append(fetch_document(client, cache, company.cik10, release["accn"], name, refresh))
                except Exception as exc:  # one failure must not stop the rest
                    row["evidence"] = f"could not fetch the release: {type(exc).__name__}: {exc}"
            if cache.filing_path(company.cik10, release["accn"], name).exists():
                exhibit = press_release(parse_submission(read_document(cache, company.cik10, release["accn"], name)))
                if exhibit is None:
                    row["evidence"] = "the 8-K has no press release exhibit to check"
                else:
                    row["status"], row["evidence"] = detect_deal(html_to_lines(exhibit["text"]), company.name)
            elif not row["evidence"]:
                row["evidence"] = "earnings release not downloaded yet"
        if company.ticker in manual.index:
            override = manual.loc[company.ticker]
            row.update(status=override["status"], source="manual", evidence=override["note"] or row["evidence"])
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS), records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download the press releases")
    args = parser.parse_args(argv)
    try:
        client = SecClient(config.sec_user_agent())
    except config.ConfigError as exc:
        print(f"Configuration problem: {exc}", file=sys.stderr)
        return 2
    cache = Cache(config.data_dir())
    try:
        scores = read_table(cache, "scores", "scores")
        companies = read_table(cache, "companies")
        filings = read_table(cache, "filings")
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    as_of = pd.Timestamp(scores["as_of"].iloc[0])
    universe = companies[companies["ticker"].isin(scores["ticker"])][["cik10", "ticker", "name"]]
    deals, records = detect_all(client, cache, filings, universe, as_of, load_overrides(), args.refresh)
    write_manifest(cache, records)
    write_table(cache, "deals", deals, AREA)

    downloaded = sum(1 for r in records if r.status == "downloaded")
    print(f"Checked the latest earnings release of {len(deals)} companies as of {as_of:%Y-%m-%d} "
          f"({downloaded} downloaded, the rest cached)\n")
    flagged = deals[deals["status"] != "none"].sort_values(["status", "ticker"])
    for row in flagged.itertuples():
        when = "" if pd.isna(row.filed) else f" (release filed {row.filed:%Y-%m-%d})"
        print(f"{row.ticker:<5} {row.status:<15} [{row.source}]{when}\n      {row.evidence}\n")
    unchecked = deals[(deals["status"] == "none") & (deals["evidence"] != "")]
    for row in unchecked.itertuples():
        print(f"{row.ticker:<5} NOT CHECKED: {row.evidence}")
    print(f"{len(flagged)} flagged, {len(deals) - len(flagged)} clear. Overrides: config/deals.csv. "
          f"Written to {cache.root / AREA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
