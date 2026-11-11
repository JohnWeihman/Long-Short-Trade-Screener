import pandas as pd
import pytest

from screener.cleaning.metrics import resolve_all
from screener.cleaning.periods import assign_periods, fiscal_years
from screener.cleaning.quarters import annual_table, quarterly_table
from screener.cleaning.tagmap import parse_tag_map

CIK = "0000000001"

# A 52-week fiscal year ending late January, like NVDA.
FY24 = ("2023-01-30", "2024-01-28")
FY25 = ("2024-01-29", "2025-01-26")
Q1 = ("2024-01-29", "2024-04-28")
Q2 = ("2024-04-29", "2024-07-28")
Q3 = ("2024-07-29", "2024-10-27")
H1 = ("2024-01-29", "2024-07-28")
M9 = ("2024-01-29", "2024-10-27")
NEXT_Q1 = ("2025-01-27", "2025-04-27")


def tag_map():
    return parse_tag_map({
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
    })


def fact(tag, period, val, *, first=None, unit="USD", form="10-Q", filed="2024-06-01", cik=CIK):
    start, end = period if isinstance(period, tuple) else (None, period)
    first = val if first is None else first
    return {
        "cik10": cik, "taxonomy": "us-gaap", "tag": tag, "unit": unit,
        "start": pd.Timestamp(start) if start else pd.NaT, "end": pd.Timestamp(end),
        "val_first": float(first), "accn_first": "A", "form_first": form, "filed_first": pd.Timestamp(filed),
        "val_latest": float(val), "accn_latest": "B", "form_latest": form, "filed_latest": pd.Timestamp(filed),
        "n_filings": 1, "n_values": 1, "restated": first != val, "scale_errors": 0, "large_disagreement": False,
    }


def k(tag, period, val, **kw):
    return fact(tag, period, val, form="10-K", filed=kw.pop("filed", "2025-03-01"), **kw)


def company_facts():
    rows = [
        # Revenue: quarters reported for Q1-Q3, plus year-to-date and the full year.
        k("Revenues", FY24, 400, filed="2024-03-01"),
        fact("Revenues", Q1, 100, first=98),
        fact("Revenues", Q2, 110), fact("Revenues", H1, 210),
        fact("Revenues", Q3, 120), fact("Revenues", M9, 330),
        k("Revenues", FY25, 460),
        fact("Revenues", NEXT_Q1, 140, filed="2025-06-01"),
        # Periods that are not a fiscal quarter, year-to-date or year.
        fact("Revenues", ("2024-03-01", "2024-09-15"), 999),
        fact("Revenues", ("2023-07-31", "2024-07-28"), 430),  # trailing twelve months in a 10-Q
        # An older tag used for one period only.
        fact("Revenue606", ("2023-01-30", "2023-04-30"), 90, filed="2023-06-01"),
        fact("Revenue606", Q1, 100.5),  # lower priority than Revenues: must lose

        fact("CostOfRevenue", Q1, 40), fact("CostOfRevenue", Q2, 44), fact("CostOfRevenue", H1, 84),
        fact("CostOfRevenue", Q3, 48), fact("CostOfRevenue", M9, 132), k("CostOfRevenue", FY25, 184),

        # Net income: no six-month figure.
        fact("NetIncomeLoss", Q1, 20), fact("NetIncomeLoss", Q2, 22), fact("NetIncomeLoss", Q3, 24),
        fact("NetIncomeLoss", M9, 66), k("NetIncomeLoss", FY25, 92), k("NetIncomeLoss", FY24, 80, filed="2024-03-01"),

        # Cash flow: year-to-date only.
        fact("OperatingCashFlow", Q1, 30), fact("OperatingCashFlow", H1, 70),
        fact("OperatingCashFlow", M9, 120), k("OperatingCashFlow", FY25, 180),

        fact("DilutedShares", Q1, 100, unit="shares"), fact("DilutedShares", Q2, 100, unit="shares"),
        fact("DilutedShares", Q3, 104, unit="shares"), fact("DilutedShares", M9, 304 / 3, unit="shares"),
        k("DilutedShares", FY25, 102, unit="shares"),
        fact("EpsDiluted", Q1, 0.20, unit="USD/shares"), fact("EpsDiluted", Q2, 0.22, unit="USD/shares"),
        fact("EpsDiluted", Q3, 0.23, unit="USD/shares"), k("EpsDiluted", FY25, 0.90, unit="USD/shares"),

        k("Cash", "2024-01-28", 500, filed="2024-03-01"),
        fact("Cash", "2024-04-28", 510), fact("Cash", "2024-07-28", 520), fact("Cash", "2024-10-27", 530),
        k("Cash", "2025-01-26", 540),
        fact("Cash", "2024-05-15", 777),  # not a quarter end
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def built():
    metric_facts = resolve_all(company_facts(), tag_map())
    assigned, years, quarters = assign_periods(metric_facts)
    quarterly = quarterly_table(assigned, quarters, tag_map())
    annual = annual_table(assigned, quarters)
    return {"mf": metric_facts, "assigned": assigned, "years": years, "quarters": quarters,
            "quarterly": quarterly, "annual": annual}


def series(table, metric, fiscal_year, column="val"):
    rows = table[(table["metric"] == metric) & (table["fiscal_year"] == fiscal_year)]
    return dict(zip(rows["quarter"], rows[column]))


# ---- tag map applied to every period ----------------------------------

def test_priority_tag_wins_and_fallback_is_per_period(built):
    revenue = built["mf"][built["mf"]["metric"] == "revenue"].set_index(["start", "end"])
    q1 = revenue.loc[(pd.Timestamp(Q1[0]), pd.Timestamp(Q1[1]))]
    old = revenue.loc[(pd.Timestamp("2023-01-30"), pd.Timestamp("2023-04-30"))]
    assert (q1["val"], q1["val_first"], q1["source"]) == (100.0, 98.0, "Revenues")
    assert (old["val"], old["source"]) == (90.0, "Revenue606")


def test_formula_metric_is_computed_for_every_period_its_terms_share(built):
    gross = built["mf"][built["mf"]["metric"] == "gross_profit"]
    by_end = {(r.start, r.end): r for r in gross.itertuples()}
    q1 = by_end[(pd.Timestamp(Q1[0]), pd.Timestamp(Q1[1]))]
    assert (q1.val, q1.val_first, q1.derived, q1.source) == (60.0, 58.0, True, "revenue - cost_of_revenue")
    assert by_end[(pd.Timestamp(FY25[0]), pd.Timestamp(FY25[1]))].val == 276.0
    # No cost of revenue for the prior year, so no gross profit either.
    assert (pd.Timestamp(FY24[0]), pd.Timestamp(FY24[1])) not in by_end
    assert q1.filed_first == pd.Timestamp("2024-06-01")


# ---- fiscal calendar ---------------------------------------------------

def test_fiscal_years_come_from_annual_filings_only(built):
    years = built["years"]
    assert years["fiscal_year"].tolist() == [2024, 2025, 2026]
    assert years["fy_end"].iloc[:2].tolist() == [pd.Timestamp(FY24[1]), pd.Timestamp(FY25[1])]
    # The twelve months to July reported in a 10-Q did not become a fiscal year.
    assert pd.Timestamp("2024-07-28") not in set(years["fy_end"].dropna())
    in_progress = years.iloc[2]
    assert (in_progress["fy_start"], in_progress["complete"]) == (pd.Timestamp("2025-01-27"), False)
    assert pd.isna(in_progress["fy_end"])


@pytest.mark.parametrize("fy_end, label", [
    ("2026-01-03", 2025),  # LSCC / SLAB: the Saturday nearest 31 December
    ("2026-01-25", 2026),  # NVDA: last Sunday of January
    ("2025-12-31", 2025),
    ("2026-09-03", 2026),  # MU
    ("2025-11-02", 2025),  # AVGO
])
def test_fiscal_year_label(fy_end, label):
    end = pd.Timestamp(fy_end)
    mf = resolve_all(pd.DataFrame([k("Revenues", (str((end - pd.Timedelta(days=363)).date()), fy_end), 1)]), tag_map())
    assert fiscal_years(mf)["fiscal_year"].iloc[0] == label


def test_spans(built):
    revenue = built["assigned"][built["assigned"]["metric"] == "revenue"]
    got = {(r.fiscal_year, r.quarter, r.span): r.val for r in revenue.itertuples()}
    assert got == {
        (2024, 1, "Q"): 90.0, (2024, 4, "FY"): 400.0,
        (2025, 1, "Q"): 100.0, (2025, 2, "Q"): 110.0, (2025, 2, "YTD"): 210.0,
        (2025, 3, "Q"): 120.0, (2025, 3, "YTD"): 330.0, (2025, 4, "FY"): 460.0,
        (2026, 1, "Q"): 140.0,
    }  # the 999 and 430 odd periods are not assigned


def test_fiscal_quarter_ends(built):
    q = built["quarters"]
    fy25 = q[q["fiscal_year"] == 2025]
    assert fy25["period_end"].tolist() == [pd.Timestamp(d) for d in ("2024-04-28", "2024-07-28", "2024-10-27", "2025-01-26")]


def test_calendar_year_company_with_uneven_quarters():
    rows = [
        k("Revenues", ("2024-01-01", "2024-12-31"), 400),
        fact("Revenues", ("2024-01-01", "2024-03-31"), 90),
        fact("Revenues", ("2024-04-01", "2024-06-30"), 100), fact("Revenues", ("2024-01-01", "2024-06-30"), 190),
        fact("Revenues", ("2024-07-01", "2024-09-30"), 100), fact("Revenues", ("2024-01-01", "2024-09-30"), 290),
    ]
    assigned, _, quarters = assign_periods(resolve_all(pd.DataFrame(rows), tag_map()))
    table = quarterly_table(assigned, quarters, tag_map())
    assert series(table, "revenue", 2024) == {1: 90.0, 2: 100.0, 3: 100.0, 4: 110.0}


def test_53_week_year_with_a_14_week_quarter():
    rows = [
        k("Revenues", ("2023-01-30", "2024-02-04"), 530),  # 371 days
        fact("Revenues", ("2023-01-30", "2023-04-30"), 130),
        fact("Revenues", ("2023-05-01", "2023-07-30"), 130),
        fact("Revenues", ("2023-07-31", "2023-10-29"), 130),
        fact("Revenues", ("2023-01-30", "2023-10-29"), 390),
    ]
    assigned, _, quarters = assign_periods(resolve_all(pd.DataFrame(rows), tag_map()))
    assert series(quarterly_table(assigned, quarters, tag_map()), "revenue", 2024) == {1: 130.0, 2: 130.0, 3: 130.0, 4: 140.0}


# ---- quarters ----------------------------------------------------------

def test_fourth_quarter_is_full_year_minus_nine_months(built):
    q = built["quarterly"]
    assert series(q, "revenue", 2025) == {1: 100.0, 2: 110.0, 3: 120.0, 4: 130.0}
    assert series(q, "revenue", 2025, "method") == {1: "reported", 2: "reported", 3: "reported", 4: "derived"}


def test_first_reported_basis_is_carried_through(built):
    assert series(built["quarterly"], "revenue", 2025, "val_first") == {1: 98.0, 2: 110.0, 3: 120.0, 4: 130.0}


def test_cash_flow_quarters_come_from_year_to_date_differences(built):
    q = built["quarterly"]
    assert series(q, "operating_cash_flow", 2025) == {1: 30.0, 2: 40.0, 3: 50.0, 4: 60.0}
    assert series(q, "operating_cash_flow", 2025, "method") == {1: "reported", 2: "derived", 3: "derived", 4: "derived"}


def test_quarter_is_not_derived_from_figures_with_different_sources():
    # Q1 is tagged Revenues, but the six-month and full-year figures exist
    # only under the fallback tag: subtracting across tags is not allowed.
    rows = [
        k("Revenues", FY24, 400, filed="2024-03-01"),
        fact("Revenues", Q1, 100),
        fact("Revenue606", H1, 205),
        fact("Revenue606", M9, 320),
        k("Revenue606", FY25, 450), k("Revenues", ("2025-01-27", "2026-01-25"), 500, filed="2026-03-01"),
    ]
    assigned, _, quarters = assign_periods(resolve_all(pd.DataFrame(rows), tag_map()))
    table = quarterly_table(assigned, quarters, tag_map())
    got = series(table, "revenue", 2025)
    assert got == {1: 100.0, 3: 115.0, 4: 130.0}  # Q2 (205 - 100) is refused; Q3 and Q4 are like-for-like
    assert series(table, "revenue", 2025, "source")[3] == "Revenue606"


def test_fourth_quarter_without_a_six_month_figure(built):
    assert series(built["quarterly"], "net_income", 2025) == {1: 20.0, 2: 22.0, 3: 24.0, 4: 26.0}


def test_metric_built_by_formula_gets_quarters_too(built):
    q = built["quarterly"]
    assert series(q, "gross_profit", 2025) == {1: 60.0, 2: 66.0, 3: 72.0, 4: 78.0}
    assert all(series(q, "gross_profit", 2025, "formula_derived").values())
    assert not any(series(q, "revenue", 2025, "formula_derived").values())


def test_weighted_average_shares_fourth_quarter(built):
    # Full-year average 102 over four quarters, nine-month average 101.33 over three:
    # Q4 = 4 x 102 - 3 x 101.33 = 104.
    shares = series(built["quarterly"], "diluted_shares", 2025)
    assert (shares[1], shares[2], shares[3]) == (100.0, 100.0, 104.0)
    assert shares[4] == pytest.approx(104.0)
    assert series(built["quarterly"], "diluted_shares", 2025, "method")[4] == "derived"


def test_fourth_quarter_eps_is_income_over_shares_not_a_subtraction(built):
    q = built["quarterly"]
    eps = series(q, "eps_diluted", 2025)
    assert (eps[1], eps[2], eps[3]) == (0.20, 0.22, 0.23)
    assert eps[4] == pytest.approx(26 / 104)
    assert series(q, "eps_diluted", 2025, "method")[4] == "ratio"
    assert series(q, "eps_diluted", 2025, "source")[4] == "net_income / diluted_shares"


def test_year_with_only_an_annual_figure_has_no_invented_quarters(built):
    assert series(built["quarterly"], "net_income", 2024) == {}
    assert series(built["quarterly"], "revenue", 2024) == {1: 90.0}


def test_balance_sheet_values_sit_on_quarter_ends(built):
    q = built["quarterly"]
    assert series(q, "cash", 2025) == {1: 510.0, 2: 520.0, 3: 530.0, 4: 540.0}
    assert series(q, "cash", 2024) == {4: 500.0}
    assert 777.0 not in set(q["val"])


def test_known_on_is_when_the_last_ingredient_was_filed(built):
    known = series(built["quarterly"], "revenue", 2025, "known_on")
    assert known[1] == pd.Timestamp("2024-06-01")
    assert known[4] == pd.Timestamp("2025-03-01")  # needs the 10-K


def test_trailing_twelve_months(built):
    q = built["quarterly"]
    assert series(q, "revenue", 2025, "val_ttm")[4] == 460.0
    assert series(q, "revenue", 2026, "val_ttm") == {1: 110.0 + 120.0 + 130.0 + 140.0}
    assert pd.isna(series(q, "revenue", 2025, "val_ttm")[3])  # only three quarters so far
    assert series(q, "diluted_shares", 2025, "val_ttm")[4] == pytest.approx(102.0)  # mean, not sum
    assert series(q, "cash", 2025, "val_ttm")[4] == 540.0  # point in time: itself


def test_ttm_needs_four_consecutive_quarters():
    rows = [
        k("Revenues", FY24, 400, filed="2024-03-01"),
        fact("Revenues", ("2023-01-30", "2023-04-30"), 90),
        fact("Revenues", Q1, 100), fact("Revenues", Q2, 110), fact("Revenues", Q3, 120),
    ]
    assigned, _, quarters = assign_periods(resolve_all(pd.DataFrame(rows), tag_map()))
    table = quarterly_table(assigned, quarters, tag_map())
    # FY24 Q1 + the next year's Q1-Q3 are four rows but not four consecutive quarters.
    assert len(table) == 4 and table["val_ttm"].isna().all()


# ---- annual ------------------------------------------------------------

def test_annual_table_uses_the_reported_full_year(built):
    annual = built["annual"].set_index(["metric", "fiscal_year"])
    assert annual.loc[("revenue", 2025), "val"] == 460.0
    assert annual.loc[("revenue", 2024), "val"] == 400.0
    assert annual.loc[("cash", 2025), "val"] == 540.0
    assert annual.loc[("gross_profit", 2025), "formula_derived"]
    assert annual.loc[("revenue", 2025), "period_end"] == pd.Timestamp(FY25[1])
    assert ("revenue", 2026) not in annual.index  # year in progress


def test_quarters_add_up_to_the_year(built):
    q, annual = built["quarterly"], built["annual"].set_index(["metric", "fiscal_year"])
    for metric in ("revenue", "cost_of_revenue", "gross_profit", "net_income", "operating_cash_flow"):
        assert sum(series(q, metric, 2025).values()) == annual.loc[(metric, 2025), "val"]
