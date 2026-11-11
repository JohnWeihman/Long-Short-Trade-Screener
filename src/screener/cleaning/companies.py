"""Company table and filing index from the raw submissions files, plus the
filer-type classification that decides who the screener can score."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from screener.data.cache import Cache, cik10, read_gz_json

FILING_COLUMNS = ["cik10", "accn", "form", "filed", "report_date", "primary_document", "file_number"]
_SEC_FIELDS = {
    "accn": "accessionNumber",
    "form": "form",
    "filed": "filingDate",
    "report_date": "reportDate",
    "primary_document": "primaryDocument",
    "file_number": "fileNumber",
}

DOMESTIC_ANNUAL = ("10-K", "10-KT")
FOREIGN_ANNUAL = ("20-F", "40-F")

# filer_type values. Only 'operating' companies are scored.
OPERATING = "operating"


def filing_index(cik: int | str, main: dict, extras: Iterable[dict] = ()) -> pd.DataFrame:
    """Every filing in a company's submissions files, newest first.

    main is CIK##########.json (recent filings under filings.recent); extras
    are the overflow files, which hold the same arrays at the top level.
    """
    blocks = [main.get("filings", {}).get("recent", {}), *extras]
    frames = []
    for block in blocks:
        n = len(block.get("accessionNumber", []))
        if not n:
            continue
        frames.append(pd.DataFrame({
            ours: block.get(theirs) or [None] * n for ours, theirs in _SEC_FIELDS.items()
        }))
    if not frames:
        return pd.DataFrame(columns=FILING_COLUMNS)
    index = pd.concat(frames, ignore_index=True).drop_duplicates(subset="accn")
    index.insert(0, "cik10", cik10(cik))
    index["filed"] = pd.to_datetime(index["filed"], errors="coerce")
    index["report_date"] = pd.to_datetime(index["report_date"].replace("", None), errors="coerce")
    index = index.sort_values(["filed", "accn"], ascending=False, kind="stable")
    return index[FILING_COLUMNS].reset_index(drop=True)


def _base_form(form: str) -> str:
    return form.removesuffix("/A")


def latest_annual_form(index: pd.DataFrame) -> tuple[str | None, pd.Timestamp | None]:
    """(form, filing date) of the most recent annual report, amendments ignored."""
    annual = index[index["form"].isin(DOMESTIC_ANNUAL + FOREIGN_ANNUAL)]
    if annual.empty:
        return None, None
    row = annual.iloc[0]  # index is newest first
    return row["form"], row["filed"]


def classify(sic: str | None, annual_form: str | None, file_numbers: Iterable[str | None] = ()) -> str:
    """Filer type from SIC code, latest annual form, and SEC file numbers.

    Order matters: a foreign bank is 'foreign', a REIT that is also a blank
    check is a 'spac'. Shell companies cannot be identified from these fields
    alone (the shell flag is a non-numeric cover-page item that companyfacts
    does not carry); SIC 6770 catches the blank-check ones.
    """
    if annual_form is None:
        return "no_annual_report"
    if _base_form(annual_form) in FOREIGN_ANNUAL:
        return "foreign"
    if any(str(n).startswith("814-") for n in file_numbers if n):
        return "bdc"  # business development companies register under 814- file numbers
    code = int(sic) if sic and str(sic).isdigit() else None
    if code is None:
        return OPERATING
    if code == 6770:
        return "spac"
    if code == 6798:
        return "reit"
    if 6000 <= code <= 6199:
        return "bank"  # depository institutions and other lenders
    if 6300 <= code <= 6411:
        return "insurer"
    return OPERATING


def company_record(main: dict, index: pd.DataFrame) -> dict:
    """One row of the company table."""
    annual_form, annual_filed = latest_annual_form(index)
    sic = str(main.get("sic") or "") or None
    filer_type = classify(sic, annual_form, index["file_number"])
    tickers = [t for t in main.get("tickers") or [] if t]
    exchanges = [e for e in main.get("exchanges") or [] if e]
    return {
        "cik10": cik10(main["cik"]),
        "name": main.get("name"),
        "ticker": tickers[0] if tickers else None,
        "tickers": ",".join(tickers),
        "exchange": exchanges[0] if exchanges else None,
        "sic": sic,
        "sic_description": main.get("sicDescription") or None,
        "fiscal_year_end": main.get("fiscalYearEnd") or None,  # MMDD as the SEC has it
        "filer_category": main.get("category") or None,
        "state_of_incorporation": main.get("stateOfIncorporation") or None,
        "annual_form": annual_form,
        "last_annual_filed": annual_filed,
        "n_filings": len(index),
        "filer_type": filer_type,
        "include": filer_type == OPERATING,
    }


def load_submissions(cache: Cache, cik: int | str) -> tuple[dict, list[dict]]:
    """(main file, overflow files) from the raw cache."""
    key = cik10(cik)
    main = read_gz_json(cache.submissions_path(f"CIK{key}.json"))
    extras = [
        read_gz_json(cache.submissions_path(entry["name"]))
        for entry in main.get("filings", {}).get("files", [])
        if cache.submissions_path(entry["name"]).exists()
    ]
    return main, extras


def build_companies(cache: Cache, ciks: Iterable[int | str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(company table, filing index) for the given CIKs."""
    records, indexes = [], []
    for cik in ciks:
        main, extras = load_submissions(cache, cik)
        index = filing_index(cik, main, extras)
        records.append(company_record(main, index))
        indexes.append(index)
    filings = pd.concat(indexes, ignore_index=True) if indexes else pd.DataFrame(columns=FILING_COLUMNS)
    return pd.DataFrame(records), filings
