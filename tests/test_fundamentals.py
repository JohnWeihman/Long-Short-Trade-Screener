import numpy as np
import pandas as pd
import pytest

from screener.cleaning.tagmap import load_tag_map
from screener.scoring.build import load_settings
from screener.scoring.fundamentals import (
    Settings,
    build_fundamentals,
    known_as_of,
    latest_balance,
    price_as_of,
    trailing,
)

CIK = "0000000001"
AS_OF = pd.Timestamp("2026-02-15")
QUARTER_ENDS = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}


def qrow(metric, fy, q, val, *, first=None, known=None, method="reported"):
    end = pd.Timestamp(f"{fy}-{QUARTER_ENDS[q]}")
    return {
        "cik10": CIK, "metric": metric, "fiscal_year": fy, "quarter": q, "period_end": end,
        "val": float(val), "val_first": float(val if first is None else first), "method": method,
        "source": "tag", "formula_derived": False,
        "known_on": pd.Timestamp(known) if known else end + pd.Timedelta(days=40), "val_ttm": np.nan,
    }


def quarterly(extra=(), skip=lambda r: False):
    """Three calendar fiscal years. Revenue per quarter: 100 (2023), 110 (2024), 132 (2025)."""
    rows = []
    for fy, revenue in ((2023, 100), (2024, 110), (2025, 132)):
        for q in (1, 2, 3, 4):
            rows += [
                qrow("revenue", fy, q, revenue),
                qrow("gross_profit", fy, q, revenue * 0.5),
                qrow("operating_income", fy, q, revenue * 0.2),
                qrow("depreciation_amortization", fy, q, 5),
                qrow("net_income", fy, q, 15),
                qrow("eps_diluted", fy, q, 0.15),
                qrow("operating_cash_flow", fy, q, 30),
                qrow("capex", fy, q, 10),
                qrow("cash", fy, q, 200), qrow("short_term_investments", fy, q, 50),
                qrow("short_term_debt", fy, q, 20), qrow("long_term_debt", fy, q, 180),
                qrow("equity", fy, q, 400 if fy < 2025 else 500),
                qrow("shares_outstanding", fy, q, 99),
            ]
    rows.append(qrow("shares_outstanding_cover", 2025, 4, 100))
    rows = [r for r in rows if not skip(r)] + list(extra)
    return pd.DataFrame(rows)


def price_frame(rows=(("2026-02-10", 49.0), ("2026-02-13", 50.0), ("2026-02-20", 60.0))):
    return pd.DataFrame({"date": pd.to_datetime([d for d, _ in rows]), "close": [c for _, c in rows]})


COMPANIES = pd.DataFrame([{"cik10": CIK, "ticker": "TEST"}])
PEERS = pd.DataFrame([{"ticker": "TEST", "peer_group": "g"}])


def run(table=None, as_of=AS_OF, prices=None, peers=PEERS, basis="latest", settings=Settings()):
    frame = build_fundamentals(
        quarterly() if table is None else table, COMPANIES, peers, load_tag_map(),
        {"TEST": price_frame()} if prices is None else prices, as_of, settings, basis,
    )
    return frame.iloc[0]


# ---- small pieces ------------------------------------------------------

def test_trailing_needs_four_consecutive_quarters():
    series = pd.Series({10: 1.0, 11: 2.0, 12: 3.0, 13: 4.0, 15: 5.0})
    assert trailing(series, 13) == 10.0
    assert trailing(series, 13, average=True) == 2.5
    assert np.isnan(trailing(series, 15))  # 14 is missing
    assert np.isnan(trailing(series, 12))  # only three before it


def test_latest_balance_falls_back_a_limited_number_of_quarters():
    series = pd.Series({10: 7.0})
    assert latest_balance(series, 10, 2) == (7.0, 0)
    assert latest_balance(series, 12, 2) == (7.0, 2)
    value, lag = latest_balance(series, 13, 2)
    assert np.isnan(value) and lag is None


def test_price_as_of_never_looks_ahead():
    assert price_as_of(price_frame(), AS_OF) == (50.0, pd.Timestamp("2026-02-13"))
    assert price_as_of(price_frame(), pd.Timestamp("2026-02-20"))[0] == 60.0
    price, date = price_as_of(price_frame(), pd.Timestamp("2026-01-01"))
    assert np.isnan(price) and date is None
    assert np.isnan(price_as_of(None, AS_OF)[0])


def test_known_as_of_filters_on_filing_date_and_picks_the_basis():
    table = quarterly()
    assert known_as_of(table, pd.Timestamp("2025-06-01"))["known_on"].max() <= pd.Timestamp("2025-06-01")
    table.loc[0, "val_first"] = 90.0
    assert known_as_of(table, AS_OF, "first")["value"].iloc[0] == 90.0
    assert known_as_of(table, AS_OF, "latest")["value"].iloc[0] == 100.0
    with pytest.raises(ValueError):
        known_as_of(table, AS_OF, "restated")


def test_settings_file_loads():
    settings = load_settings()
    assert settings.tax_rate == 0.21 and settings.max_price_age_days == 7


# ---- one company, end to end -------------------------------------------

def test_latest_quarter_and_trailing_twelve_months():
    f = run()
    assert (f["fiscal_year"], f["quarter"], f["period_end"]) == (2025, 4, pd.Timestamp("2025-12-31"))
    assert f["known_on"] == pd.Timestamp("2026-02-09")
    assert (f["revenue_ttm"], f["revenue_ttm_1y_ago"], f["revenue_ttm_2y_ago"]) == (528.0, 440.0, 400.0)
    assert f["operating_income_ttm"] == pytest.approx(105.6)
    assert f["operating_income_ttm_1y_ago"] == pytest.approx(88.0)
    assert f["eps_diluted_ttm"] == pytest.approx(0.60)
    assert f["notes"] == ""


def test_ebitda_and_free_cash_flow():
    f = run()
    assert f["ebitda_ttm"] == pytest.approx(105.6 + 20)
    assert (f["fcf_ttm"], f["fcf_ttm_1y_ago"]) == (80.0, 80.0)


def test_market_cap_net_debt_and_enterprise_value():
    f = run()
    assert (f["shares"], f["shares_source"], f["price"]) == (100.0, "shares_outstanding_cover", 50.0)
    assert f["market_cap"] == 5000.0
    assert (f["total_debt"], f["net_debt"]) == (200.0, -50.0)  # 20 + 180 - 200 - 50
    assert f["enterprise_value"] == 4950.0


def test_roic_uses_average_invested_capital_and_the_flat_tax_rate():
    f = run()
    assert (f["invested_capital"], f["invested_capital_1y_ago"]) == (450.0, 350.0)
    assert f["nopat_ttm"] == pytest.approx(105.6 * 0.79)
    assert f["roic"] == pytest.approx(105.6 * 0.79 / 400)
    assert run(settings=Settings(tax_rate=0.0))["roic"] == pytest.approx(105.6 / 400)


def test_as_of_uses_only_what_had_been_filed():
    # Q4 2025 results were filed 2026-02-09; a week earlier the latest quarter is Q3.
    f = run(as_of=pd.Timestamp("2026-02-01"), prices={"TEST": price_frame((("2026-01-30", 40.0),))})
    assert (f["fiscal_year"], f["quarter"]) == (2025, 3)
    assert f["revenue_ttm"] == 110 + 132 * 3
    assert f["shares_source"] == "shares_outstanding"  # the cover count came with the Q4 filing
    assert f["market_cap"] == 99 * 40.0


def test_first_reported_basis():
    table = quarterly()
    table.loc[(table["metric"] == "revenue") & (table["fiscal_year"] == 2025), "val_first"] = 130.0
    assert run(table, basis="first")["revenue_ttm"] == 520.0
    assert run(table, basis="latest")["revenue_ttm"] == 528.0


def test_missing_depreciation_blanks_ebitda_but_not_the_rest():
    f = run(quarterly(skip=lambda r: r["metric"] == "depreciation_amortization" and r["fiscal_year"] == 2025 and r["quarter"] == 2))
    assert np.isnan(f["ebitda_ttm"]) and f["fcf_ttm"] == 80.0
    assert "no D&A" in f["notes"]


def test_unreported_debt_is_zero_and_noted():
    f = run(quarterly(skip=lambda r: r["metric"] in ("short_term_debt", "short_term_investments")))
    assert (f["short_term_debt"], f["short_term_investments"]) == (0.0, 0.0)
    assert f["net_debt"] == 180.0 - 200.0
    assert "short_term_debt not reported: taken as 0" in f["notes"]


def test_stale_balance_sheet_value_is_used_and_noted():
    f = run(quarterly(skip=lambda r: r["metric"] == "cash" and r["fiscal_year"] == 2025 and r["quarter"] == 4))
    assert f["cash"] == 200.0 and "cash is 1 quarter(s) old" in f["notes"]


def test_stale_price_and_missing_price_are_noted():
    old = run(prices={"TEST": price_frame((("2026-01-02", 45.0),))})
    assert old["price"] == 45.0 and "price is 44 days old" in old["notes"]
    none = run(prices={})
    assert np.isnan(none["market_cap"]) and "no price" in none["notes"]


def test_negative_invested_capital_gives_no_roic():
    table = quarterly()
    table.loc[table["metric"] == "equity", "val"] = -100.0
    f = run(table)
    assert np.isnan(f["roic"]) and "ROIC undefined" in f["notes"]


def test_company_without_enough_history():
    table = quarterly(skip=lambda r: r["fiscal_year"] < 2025 or r["quarter"] < 3)
    f = run(table)
    assert np.isnan(f["revenue_ttm"]) and "fewer than four consecutive quarters" in f["notes"]


def test_companies_missing_from_clean_data_or_unplaced_still_get_a_row():
    peers = pd.DataFrame([{"ticker": "TEST", "peer_group": None}, {"ticker": "GONE", "peer_group": "g"}])
    frame = build_fundamentals(quarterly(), COMPANIES, peers, load_tag_map(), {"TEST": price_frame()}, AS_OF)
    assert frame["ticker"].tolist() == ["TEST", "GONE"]
    assert frame.loc[0, "peer_group"] is None and frame.loc[0, "market_cap"] == 5000.0
    assert frame.loc[1, "notes"] == "not in the cleaned company table"
