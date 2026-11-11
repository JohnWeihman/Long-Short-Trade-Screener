import json

import pandas as pd
import pytest

from screener.cleaning.build import build
from screener.cleaning.facts import (
    FACT_COLUMNS,
    RAW_COLUMNS,
    dedupe,
    flatten_company,
    is_statement_form,
    mapped_tags,
)
from screener.cleaning.store import read_table
from screener.cleaning.tagmap import parse_tag_map
from screener.data.cache import write_gz

from test_companies import RECENT, submissions
from test_rawfacts import company

CIK = "0000001234"
REV = "RevenueFromContractWithCustomerExcludingAssessedTax"


def tag_map():
    return parse_tag_map({
        "revenue": {"kind": "duration", "unit": "USD", "tags": ["Revenues", REV]},
        "net_income": {"kind": "duration", "unit": "USD", "tags": ["NetIncomeLoss"]},
        "shares_outstanding": {"kind": "instant", "unit": "shares", "tags": ["CommonStockSharesOutstanding"]},
        "shares_outstanding_cover": {"kind": "cover", "unit": "shares", "tags": ["dei:EntityCommonStockSharesOutstanding"]},
        "gross_profit": {"kind": "duration", "unit": "USD", "formulas": [
            {"add": ["revenue"], "subtract": ["CostOfRevenue"], "add_if_present": ["NetIncomeLoss"]},
        ]},
    })


def raw():
    return flatten_company(company(), CIK, mapped_tags(tag_map()))


def one(facts, tag, end, start=None):
    rows = facts[(facts["tag"] == tag) & (facts["end"] == pd.Timestamp(end))]
    rows = rows[rows["start"].isna()] if start is None else rows[rows["start"] == pd.Timestamp(start)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_mapped_tags_include_formula_tags_once_and_skip_metric_terms():
    assert mapped_tags(tag_map()) == [
        ("us-gaap", "Revenues"),
        ("us-gaap", REV),
        ("us-gaap", "NetIncomeLoss"),
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CostOfRevenue"),
    ]


def test_flatten_keeps_every_reported_row_of_mapped_tags_only():
    frame = raw()
    assert list(frame.columns) == RAW_COLUMNS
    assert len(frame) == 1 + 6 + 2 + 2 + 3  # Revenues, ASC 606 revenue, net income, shares, cover
    assert "ProfitLoss" not in set(frame["tag"])  # in the data, not in this map
    assert set(frame["cik10"]) == {CIK}
    assert frame["end"].dtype.kind == "M" and frame["val"].dtype == "float64"
    assert frame[frame["tag"] == "CommonStockSharesOutstanding"]["start"].isna().all()


def test_dedupe_gives_one_row_per_period():
    facts = dedupe(raw())
    assert list(facts.columns) == FACT_COLUMNS
    assert not facts.duplicated(["cik10", "taxonomy", "tag", "unit", "start", "end"]).any()
    assert len(facts) == 1 + 5 + 1 + 2 + 3  # prior-year revenue and net income each collapse


def test_same_value_in_two_filings_is_not_a_restatement():
    row = one(dedupe(raw()), REV, "2024-01-28", "2023-01-30")
    assert (row["val_first"], row["val_latest"]) == (400.0, 400.0)
    assert (row["accn_first"], row["accn_latest"]) == ("A-24", "0000000001-25-000010")
    assert (row["n_filings"], row["n_values"], row["restated"]) == (2, 1, False)


def test_restatement_keeps_both_first_reported_and_latest():
    row = one(dedupe(raw()), "NetIncomeLoss", "2025-01-26", "2024-01-29")
    assert (row["val_first"], row["form_first"], row["filed_first"]) == (90.0, "10-K", pd.Timestamp("2025-02-20"))
    assert (row["val_latest"], row["form_latest"], row["filed_latest"]) == (85.0, "10-K/A", pd.Timestamp("2025-06-01"))
    assert (row["n_filings"], row["n_values"], row["restated"]) == (2, 2, True)


def test_proxy_statement_values_never_count():
    # A proxy filed after the 10-K/A re-tags net income with a different figure.
    frame = raw()
    proxy = frame[(frame["tag"] == "NetIncomeLoss") & (frame["form"] == "10-K/A")].copy()
    proxy[["val", "form", "filed", "accn"]] = [70.0, "DEF 14A", pd.Timestamp("2025-09-01"), "PX-25"]
    only_in_proxy = proxy.copy()
    only_in_proxy[["start", "end"]] = [pd.Timestamp("2022-01-31"), pd.Timestamp("2023-01-29")]
    facts = dedupe(pd.concat([frame, proxy, only_in_proxy], ignore_index=True))

    row = one(facts, "NetIncomeLoss", "2025-01-26", "2024-01-29")
    assert (row["val_latest"], row["form_latest"], row["n_filings"]) == (85.0, "10-K/A", 2)
    # A period that appears only in a proxy is not a fact at all.
    assert facts[(facts["tag"] == "NetIncomeLoss") & (facts["end"] == pd.Timestamp("2023-01-29"))].empty


@pytest.mark.parametrize("form, expected", [
    ("10-K", True), ("10-K/A", True), ("10-Q", True), ("8-K", True), ("8-K/A", True),
    ("20-F", True), ("S-1/A", True),
    ("DEF 14A", False), ("PRE 14A", False), ("DEFA14A", False), (None, False),
])
def test_is_statement_form(form, expected):
    assert is_statement_form(form) is expected


def test_first_and_latest_do_not_depend_on_input_order():
    shuffled = raw().sample(frac=1, random_state=7)
    row = one(dedupe(shuffled), "NetIncomeLoss", "2025-01-26", "2024-01-29")
    assert (row["val_first"], row["val_latest"]) == (90.0, 85.0)


def test_periods_with_the_same_end_but_different_lengths_stay_separate():
    facts = dedupe(raw())
    assert one(facts, REV, "2025-01-26", "2024-01-29")["val_latest"] == 500.0  # full year
    assert one(facts, REV, "2025-01-26", "2024-10-28")["val_latest"] == 140.0  # fourth quarter


def test_point_in_time_facts_survive_dedupe():
    facts = dedupe(raw())
    assert one(facts, "CommonStockSharesOutstanding", "2025-01-26")["val_latest"] == 990.0
    assert one(facts, "EntityCommonStockSharesOutstanding", "2025-02-14")["taxonomy"] == "dei"


def test_dedupe_of_nothing():
    empty = flatten_company({"facts": {}}, CIK, mapped_tags(tag_map()))
    assert empty.empty and list(dedupe(empty).columns) == FACT_COLUMNS


def test_build_writes_all_tables(cache):
    write_gz(cache.companyfacts_path(CIK), json.dumps(company()).encode())
    write_gz(cache.submissions_path(f"CIK{CIK}.json"), json.dumps(submissions(RECENT, cik="1234")).encode())
    tables = build(cache, [CIK], tag_map())
    assert set(tables) == {
        "companies", "filings", "splits", "facts_raw", "facts", "metric_facts",
        "fiscal_years", "fiscal_quarters", "metrics_quarterly", "metrics_annual", "validation",
    }
    for name in tables:
        assert len(read_table(cache, name)) == len(tables[name])
    saved = read_table(cache, "facts")
    assert len(saved) == len(tables["facts"]) == 12
    assert read_table(cache, "companies").loc[0, "cik10"] == CIK
    assert pd.api.types.is_datetime64_any_dtype(saved["end"])
