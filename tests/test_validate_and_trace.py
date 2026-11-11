import pandas as pd
import pytest

from screener.cleaning import validate as v
from screener.cleaning.metrics import resolve_all
from screener.cleaning.periods import assign_periods
from screener.cleaning.quarters import annual_table, quarterly_table
from screener.cleaning.tagmap import parse_tag_map
from screener.cleaning.trace import leaf_tags, periods_behind, raw_behind

from test_periods_and_quarters import CIK, FY24, FY25, M9, Q1, Q2, Q3, company_facts, fact, k

EXTRA = {
    "total_assets": {"kind": "instant", "unit": "USD", "tags": ["Assets"]},
    "total_liabilities": {"kind": "instant", "unit": "USD", "tags": ["Liabilities"]},
    "equity": {"kind": "instant", "unit": "USD", "tags": ["StockholdersEquity"]},
    "shares_outstanding": {"kind": "instant", "unit": "shares", "tags": ["CommonStockSharesOutstanding"]},
    "shares_outstanding_cover": {"kind": "cover", "unit": "shares", "tags": ["dei:EntityCommonStockSharesOutstanding"]},
    "buybacks": {"kind": "duration", "unit": "USD", "optional": True, "tags": ["Buybacks"]},
}


@pytest.fixture(autouse=True)
def small_numbers(monkeypatch):
    # The test company's figures are in the hundreds; real ones are in millions.
    monkeypatch.setattr(v, "MIN_DOLLARS", 0.0)


def tag_map():
    from test_periods_and_quarters import tag_map as base
    raw = {
        "revenue": {"kind": "duration", "unit": "USD", "tags": ["Revenues", "Revenue606"]},
        "cost_of_revenue": {"kind": "duration", "unit": "USD", "tags": ["CostOfRevenue"]},
        "gross_profit": {"kind": "duration", "unit": "USD", "tags": ["GrossProfit"],
                         "formulas": [{"add": ["revenue"], "subtract": ["cost_of_revenue"]}]},
        "net_income": {"kind": "duration", "unit": "USD", "tags": ["NetIncomeLoss"]},
        "operating_cash_flow": {"kind": "duration", "unit": "USD", "tags": ["OperatingCashFlow"]},
        "diluted_shares": {"kind": "duration", "unit": "shares", "combine": "average", "tags": ["DilutedShares"]},
        "eps_diluted": {"kind": "duration", "unit": "USD/shares", "combine": "ratio",
                        "ratio_of": ["net_income", "diluted_shares"], "tags": ["EpsDiluted"]},
        "cash": {"kind": "instant", "unit": "USD", "tags": ["Cash"]},
    }
    assert set(raw) == set(base())  # stays in step with the shared fixture
    raw.update(EXTRA)
    return parse_tag_map(raw)


def build(extra_rows=(), drop=lambda r: False):
    """Cleaned tables for the shared test company, plus/minus some facts."""
    rows = [r for r in company_facts().to_dict("records") if not drop(r)] + list(extra_rows)
    facts = pd.DataFrame(rows)
    tm = tag_map()
    metric_facts = resolve_all(facts, tm)
    covers = {n for n, m in tm.items() if m.kind == "cover"}
    assigned, years, quarters = assign_periods(metric_facts, covers)
    raw = facts.rename(columns={"val_latest": "val", "accn_latest": "accn", "form_latest": "form", "filed_latest": "filed"})
    raw = raw.assign(val_reported=raw["val"], split_factor=1.0, scale_suspect=False)
    tables = {
        "companies": pd.DataFrame([{"cik10": CIK, "ticker": "TEST", "name": "Test Semi"}]),
        "facts": facts, "facts_raw": raw, "metric_facts": metric_facts,
        "fiscal_years": years, "fiscal_quarters": quarters,
        "metrics_quarterly": quarterly_table(assigned, quarters, tm),
        "metrics_annual": annual_table(assigned, quarters),
    }
    tables["validation"] = v.validate(tables, tm)
    return tables


BALANCED = [
    k("Assets", "2025-01-26", 1000), k("Liabilities", "2025-01-26", 400), k("StockholdersEquity", "2025-01-26", 600),
    k("CommonStockSharesOutstanding", "2025-01-26", 100, unit="shares"),
]


def checks(tables, check):
    found = tables["validation"]
    return found[found["check"] == check]


def test_clean_company_has_no_errors_or_identity_warnings():
    found = build(BALANCED)["validation"]
    assert not (found["severity"] == "error").any()
    assert set(found["check"]) <= {"missing_annual", "missing_quarters"}
    assert list(found.columns) == v.COLUMNS and (found["ticker"] == "TEST").all()


def test_quarters_that_do_not_sum_to_the_year_are_flagged():
    # Reported Q1-Q3 (330) disagree with a restated nine-month figure (300).
    tables = build([fact("Revenues", M9, 300)], drop=lambda r: r["tag"] == "Revenues" and r["start"] == pd.Timestamp(M9[0]) and r["end"] == pd.Timestamp(M9[1]))
    found = checks(tables, "quarters_vs_year")
    row = found[found["metric"] == "revenue"].iloc[0]
    assert (row["fiscal_year"], row["value"], row["expected"], row["severity"]) == (2025, 490.0, 460.0, "warning")


def test_negative_derived_quarter_is_an_error():
    # The year was later re-presented at 300 but the nine-month figure stayed at 330.
    is_fy = lambda r: r["tag"] == "Revenues" and r["start"] == pd.Timestamp(FY25[0]) and r["end"] == pd.Timestamp(FY25[1])  # noqa: E731
    tables = build([k("Revenues", FY25, 300, first=460)], drop=is_fy)
    negative = checks(tables, "negative_value")
    assert negative[["metric", "fiscal_year", "quarter", "value", "severity"]].values.tolist() == [["revenue", 2025, 4, -30.0, "error"]]
    recast = checks(tables, "recast_in_derived_quarter")
    row = recast[recast["metric"] == "revenue"].iloc[0]
    assert (row["value"], row["expected"]) == (-30.0, 130.0)


def test_gross_profit_identity_only_checks_reported_gross_profit():
    assert checks(build(BALANCED), "gross_profit_identity").empty  # formula-built: equal by construction
    tables = build(BALANCED + [k("GrossProfit", FY25, 250)])
    row = checks(tables, "gross_profit_identity").iloc[0]
    assert (row["value"], row["expected"], row["fiscal_year"]) == (250.0, 276.0, 2025)


def test_balance_sheet_identity():
    off = [k("Assets", "2025-01-26", 1000), k("Liabilities", "2025-01-26", 400), k("StockholdersEquity", "2025-01-26", 500)]
    row = checks(build(off), "balance_sheet_identity").iloc[0]
    assert (row["value"], row["expected"]) == (1000.0, 900.0)
    within = [k("Assets", "2025-01-26", 1000), k("Liabilities", "2025-01-26", 400), k("StockholdersEquity", "2025-01-26", 570)]
    assert checks(build(within), "balance_sheet_identity").empty  # 3%: minority interests


def test_year_over_year_jump():
    tables = build([k("Assets", "2024-01-28", 900, filed="2024-03-01"), k("Assets", "2025-01-26", 900_000)])
    row = checks(tables, "year_over_year_jump").iloc[0]
    assert (row["metric"], row["fiscal_year"], row["value"], row["expected"]) == ("total_assets", 2025, 900_000.0, 900.0)
    assert checks(build(BALANCED), "year_over_year_jump").empty  # revenue 400 -> 460 is fine


def test_q4_eps_check():
    # Annual EPS 0.90 less 0.65 for Q1-Q3 = 0.25; income / shares gives 26/104 = 0.25: no finding.
    assert checks(build(BALANCED), "q4_eps").empty
    is_eps_fy = lambda r: r["tag"] == "EpsDiluted" and r["start"] == pd.Timestamp(FY25[0]) and r["end"] == pd.Timestamp(FY25[1])  # noqa: E731
    tables = build([k("EpsDiluted", FY25, 0.80, unit="USD/shares")], drop=is_eps_fy)
    row = checks(tables, "q4_eps").iloc[0]
    assert row["value"] == pytest.approx(0.25) and row["expected"] == pytest.approx(0.15)


def test_filing_conflicts_and_scale_fixes_are_reported():
    conflict = {**k("StockholdersEquity", "2025-01-26", 11, first=600), "large_disagreement": True}
    fixed = {**k("Assets", "2025-01-26", 1000), "scale_errors": 1}
    old = {**k("Assets", "2010-01-31", 5), "large_disagreement": True}  # outside the recent window
    tables = build([conflict, fixed, old, k("Liabilities", "2025-01-26", 989)])
    row = checks(tables, "filings_disagree")
    assert row[["metric", "value", "expected"]].values.tolist() == [["StockholdersEquity", 11.0, 600.0]]
    assert checks(tables, "scale_error_set_aside")["metric"].tolist() == ["Assets"]


def test_coverage_missing_annual_and_missing_quarters():
    tables = build(BALANCED)
    missing = checks(tables, "missing_annual")
    # FY2025 has everything required; FY2024 lacks most of it.
    assert 2025 not in set(missing["fiscal_year"])
    assert {"cost_of_revenue", "total_assets"} <= set(missing[missing["fiscal_year"] == 2024]["metric"])
    assert "buybacks" not in set(missing["metric"])  # optional
    partial = checks(tables, "missing_quarters")
    assert ("total_assets", 2025) in set(zip(partial["metric"], partial["fiscal_year"]))  # year end only


def test_one_share_count_is_enough():
    balance_only = build(BALANCED)
    assert not {"shares_outstanding", "shares_outstanding_cover"} & set(
        checks(balance_only, "missing_annual").query("fiscal_year == 2025")["metric"])
    neither = build([r for r in BALANCED if r["tag"] != "CommonStockSharesOutstanding"])
    assert {"shares_outstanding", "shares_outstanding_cover"} <= set(
        checks(neither, "missing_annual").query("fiscal_year == 2025")["metric"])


def test_cover_page_count_is_assigned_to_the_quarter_before_it():
    cover = {**k("EntityCommonStockSharesOutstanding", "2025-02-20", 101, unit="shares"), "taxonomy": "dei"}
    stale = {**k("EntityCommonStockSharesOutstanding", "2025-12-20", 999, unit="shares"), "taxonomy": "dei"}  # >150 days after any quarter end
    tables = build(BALANCED + [cover, stale])
    q = tables["metrics_quarterly"]
    rows = q[q["metric"] == "shares_outstanding_cover"]
    assert rows[["fiscal_year", "quarter", "val"]].values.tolist() == [[2025, 4, 101.0]]
    annual = tables["metrics_annual"]
    assert annual[annual["metric"] == "shares_outstanding_cover"]["val"].tolist() == [101.0]


def test_summary_counts_findings_and_companies():
    summary = v.summary(build(BALANCED)["validation"])
    assert set(summary.columns) == {"severity", "check", "findings", "companies"}
    assert (summary["companies"] == 1).all()
    assert v.summary(pd.DataFrame(columns=v.COLUMNS)).empty


# ---- trace -------------------------------------------------------------

def test_trace_reported_quarter_points_at_its_one_period():
    tables = build(BALANCED)
    periods = periods_behind(tables, tag_map(), CIK, "revenue", 2025, 2)
    assert periods[["start", "end", "val"]].values.tolist() == [
        [pd.Timestamp(Q1[0]), pd.Timestamp(Q2[1]), 210.0],  # the six-month figure ends the same day
        [pd.Timestamp(Q2[0]), pd.Timestamp(Q2[1]), 110.0],
    ]


def test_trace_derived_quarter_includes_the_year_to_date_subtracted():
    tables = build(BALANCED)
    periods = periods_behind(tables, tag_map(), CIK, "revenue", 2025, 4)
    assert sorted(periods["val"].tolist()) == [120.0, 330.0, 460.0]  # Q3, nine months, full year
    raw = raw_behind(tables, tag_map(), CIK, "revenue", periods)
    assert set(raw["tag"]) == {"Revenues"} and len(raw) == 3


def test_trace_annual_is_the_full_year_period_only():
    tables = build(BALANCED)
    periods = periods_behind(tables, tag_map(), CIK, "revenue", 2025, None)
    assert periods[["start", "end", "val"]].values.tolist() == [[pd.Timestamp(FY25[0]), pd.Timestamp(FY25[1]), 460.0]]
    assert periods_behind(tables, tag_map(), CIK, "revenue", 1999, None).empty


def test_trace_formula_metric_reaches_the_underlying_tags():
    tables = build(BALANCED)
    tm = tag_map()
    assert leaf_tags(tables["metric_facts"], tm, CIK, "gross_profit", pd.Timestamp(Q3[0]), pd.Timestamp(Q3[1])) == [
        "Revenues", "CostOfRevenue"]
    periods = periods_behind(tables, tm, CIK, "gross_profit", 2025, None)
    raw = raw_behind(tables, tm, CIK, "gross_profit", periods)
    assert sorted(raw["tag"]) == ["CostOfRevenue", "Revenues"]
    assert sorted(raw["val"]) == [184.0, 460.0]


def test_trace_balance_sheet_value():
    tables = build(BALANCED)
    periods = periods_behind(tables, tag_map(), CIK, "cash", 2025, 3)
    assert periods[["end", "val"]].values.tolist() == [[pd.Timestamp("2024-10-27"), 530.0]]
    assert pd.isna(periods["start"].iloc[0])
