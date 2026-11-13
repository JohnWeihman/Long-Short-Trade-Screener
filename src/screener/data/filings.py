"""Download filing documents (10-K main document, 8-K earnings press
releases) from EDGAR into the cache, and pick which filings to fetch.

Fetching only: turning the documents into text is memos/extract.py.
"""

from __future__ import annotations

import re

import pandas as pd

from screener.data.cache import BytesFetcher, Cache, FetchRecord, cik10, fetch_to_cache, read_gz

ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{name}"
EARNINGS_ITEM = "2.02"  # 8-K item: Results of Operations and Financial Condition

_DOCUMENT = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.S | re.I)
_FIELD = {name: re.compile(rf"<{name}>([^\r\n<]*)", re.I) for name in ("TYPE", "FILENAME", "DESCRIPTION")}
_TEXT = re.compile(r"<TEXT>(.*?)(?:</TEXT>|\Z)", re.S | re.I)


def document_url(cik: int | str, accession: str, file_name: str) -> str:
    return ARCHIVE_URL.format(cik=int(cik10(cik)), folder=accession.replace("-", ""), name=file_name)


def fetch_document(
    client: BytesFetcher, cache: Cache, cik: int | str, accession: str, file_name: str, refresh: bool = False
) -> FetchRecord:
    """One file of a filing, cached. file_name '<accession>.txt' is the
    complete submission: every document of the filing in one text file."""
    return fetch_to_cache(
        client, document_url(cik, accession, file_name), cache.filing_path(cik, accession, file_name),
        source="edgar/filings", key=f"{cik10(cik)}/{accession}/{file_name}", refresh=refresh,
    )


def submission_file_name(accession: str) -> str:
    return f"{accession}.txt"


def read_document(cache: Cache, cik: int | str, accession: str, file_name: str) -> str:
    return read_gz(cache.filing_path(cik, accession, file_name)).decode("utf-8", errors="replace")


def latest_annual_report(filings: pd.DataFrame, cik: int | str, as_of: pd.Timestamp) -> pd.Series | None:
    """The most recent 10-K (not an amendment) filed on or before as_of."""
    rows = filings[(filings["cik10"] == cik10(cik)) & (filings["form"] == "10-K") & (filings["filed"] <= as_of)]
    rows = rows[rows["primary_document"].notna() & (rows["primary_document"] != "")]
    return None if rows.empty else rows.sort_values(["filed", "accn"]).iloc[-1]


def earnings_releases(filings: pd.DataFrame, cik: int | str, as_of: pd.Timestamp, count: int) -> pd.DataFrame:
    """The latest `count` 8-Ks reporting results (item 2.02), newest first."""
    rows = filings[(filings["cik10"] == cik10(cik)) & (filings["form"] == "8-K") & (filings["filed"] <= as_of)]
    items = rows["items"].fillna("").astype(str)
    is_earnings = items.map(lambda text: EARNINGS_ITEM in [part.strip() for part in text.split(",")])
    # astype(bool): for a company with no 8-Ks the mask is empty and would
    # otherwise be read as a column selection.
    return rows[is_earnings.astype(bool)].sort_values(["filed", "accn"], ascending=False).head(count)


def parse_submission(text: str) -> list[dict]:
    """The documents inside a complete-submission text file:
    [{'type', 'file_name', 'description', 'text'}, ...] in filing order."""
    documents = []
    for block in _DOCUMENT.findall(text):
        body = _TEXT.search(block)
        field = lambda name: (m.group(1).strip() if (m := _FIELD[name].search(block)) else "")  # noqa: E731
        documents.append({
            "type": field("TYPE"), "file_name": field("FILENAME"),
            "description": field("DESCRIPTION"), "text": body.group(1) if body else "",
        })
    return documents


def press_release(documents: list[dict]) -> dict | None:
    """The earnings press release exhibit: EX-99.1, else the first EX-99.*."""
    exhibits = [d for d in documents if d["type"].upper().startswith("EX-99")]
    for document in exhibits:
        if document["type"].upper() in ("EX-99.1", "EX-99.01"):
            return document
    return exhibits[0] if exhibits else None
