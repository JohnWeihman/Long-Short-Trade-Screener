import pandas as pd
import pytest

from screener.cleaning.companies import (
    build_companies,
    classify,
    company_record,
    filing_index,
    latest_annual_form,
)
from screener.data.cache import write_gz

import json


def block(rows):
    """rows: (accession, form, filed, report_date, file_number) -> SEC column arrays."""
    return {
        "accessionNumber": [r[0] for r in rows],
        "form": [r[1] for r in rows],
        "filingDate": [r[2] for r in rows],
        "reportDate": [r[3] for r in rows],
        "primaryDocument": [f"{r[0]}.htm" for r in rows],
        "fileNumber": [r[4] for r in rows],
    }


def submissions(rows, extra_files=(), **overrides):
    main = {
        "cik": "1045810",
        "name": "NVIDIA CORP",
        "tickers": ["NVDA"],
        "exchanges": ["Nasdaq"],
        "sic": "3674",
        "sicDescription": "Semiconductors & Related Devices",
        "fiscalYearEnd": "0125",
        "category": "Large accelerated filer",
        "stateOfIncorporation": "DE",
        "filings": {"recent": block(rows), "files": [{"name": n} for n in extra_files]},
    }
    main.update(overrides)
    return main


RECENT = [
    ("A-3", "10-Q", "2025-05-28", "2025-04-27", "000-23985"),
    ("A-2", "10-K", "2025-02-26", "2025-01-26", "000-23985"),
    ("A-1", "8-K", "2025-02-26", "", "000-23985"),
]
OLDER = [
    ("A-2", "10-K", "2025-02-26", "2025-01-26", "000-23985"),  # overlap with recent
    ("A-0", "10-K", "2024-02-21", "2024-01-28", "000-23985"),
]


def test_filing_index_merges_overflow_newest_first_without_duplicates():
    index = filing_index(1045810, submissions(RECENT), [block(OLDER)])
    assert index["accn"].tolist() == ["A-3", "A-2", "A-1", "A-0"]
    assert index["cik10"].unique().tolist() == ["0001045810"]
    assert index.loc[0, "filed"] == pd.Timestamp("2025-05-28")
    assert pd.isna(index.loc[2, "report_date"])  # 8-K has no report date


def test_filing_index_of_a_company_with_no_filings():
    assert filing_index(1, {"filings": {"recent": {}}}).empty


def test_latest_annual_form_ignores_amendments_and_other_forms():
    rows = [
        ("B-3", "10-K/A", "2025-04-01", "2025-01-26", None),
        ("B-2", "10-K", "2025-02-26", "2025-01-26", None),
        ("B-1", "20-F", "2018-03-01", "2017-12-31", None),
    ]
    form, filed = latest_annual_form(filing_index(1, submissions(rows)))
    assert (form, filed) == ("10-K", pd.Timestamp("2025-02-26"))


@pytest.mark.parametrize("sic, annual_form, file_numbers, expected", [
    ("3674", "10-K", [], "operating"),
    ("6021", "10-K", [], "bank"),
    ("6141", "10-K", [], "bank"),          # personal credit institutions
    ("6331", "10-K", [], "insurer"),
    ("6411", "10-K", [], "insurer"),
    ("6798", "10-K", [], "reit"),
    ("6770", "10-K", [], "spac"),
    ("6211", "10-K", [], "operating"),     # brokers and asset managers stay in
    ("3674", "20-F", [], "foreign"),
    ("3674", "40-F", [], "foreign"),
    ("6021", "20-F", [], "foreign"),       # foreign beats bank
    ("6799", "10-K", ["814-00123", None], "bdc"),
    ("3674", None, [], "no_annual_report"),
    (None, "10-K", [], "operating"),
    ("", "10-KT", [], "operating"),
])
def test_classify(sic, annual_form, file_numbers, expected):
    assert classify(sic, annual_form, file_numbers) == expected


def test_company_that_switched_from_20f_to_10k_is_operating():
    # NXP-style: older annual reports were 20-Fs, the latest is a 10-K.
    rows = [
        ("C-2", "10-K", "2025-02-20", "2024-12-31", None),
        ("C-1", "20-F", "2018-03-01", "2017-12-31", None),
    ]
    main = submissions(rows)
    record = company_record(main, filing_index(1045810, main))
    assert (record["annual_form"], record["filer_type"], record["include"]) == ("10-K", "operating", True)


def test_company_record_fields():
    main = submissions(RECENT)
    record = company_record(main, filing_index(1045810, main))
    assert record["cik10"] == "0001045810"
    assert (record["ticker"], record["tickers"], record["exchange"]) == ("NVDA", "NVDA", "Nasdaq")
    assert (record["sic"], record["fiscal_year_end"]) == ("3674", "0125")
    assert record["last_annual_filed"] == pd.Timestamp("2025-02-26")
    assert record["n_filings"] == 3


def test_company_with_no_ticker_or_sic_does_not_break():
    main = submissions(RECENT, tickers=[], exchanges=[], sic="")
    record = company_record(main, filing_index(1045810, main))
    assert record["ticker"] is None and record["sic"] is None and record["filer_type"] == "operating"


def test_excluded_filer_is_kept_in_the_table_but_flagged():
    main = submissions(RECENT, sic="6798")
    record = company_record(main, filing_index(1045810, main))
    assert (record["filer_type"], record["include"]) == ("reit", False)


def test_build_companies_reads_main_and_overflow_from_the_cache(cache):
    extra = "CIK0001045810-submissions-001.json"
    write_gz(cache.submissions_path("CIK0001045810.json"), json.dumps(submissions(RECENT, [extra])).encode())
    write_gz(cache.submissions_path(extra), json.dumps(block(OLDER)).encode())
    companies, filings = build_companies(cache, ["0001045810"])
    assert len(companies) == 1 and companies.loc[0, "n_filings"] == 4
    assert filings["accn"].tolist() == ["A-3", "A-2", "A-1", "A-0"]
