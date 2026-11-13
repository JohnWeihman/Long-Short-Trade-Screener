"""Peer medians, quarterly trends and the filing-only fact sheet."""

import math

import numpy as np
import pandas as pd
import pytest

from screener.data.filings import latest_relevant_filing
from screener.memos.facts import fact_sheet, filing_facts, peer_median, quarter_trends
from screener.memos.write import render_template, split_sections

from test_memo_writing import pair_row, tables

AS_OF = pd.Timestamp("2026-11-11")
ENDS = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}


def metrics_table():
    return pd.DataFrame({
        "ticker": ["AAA", "BBB", "CCC", "DDD", "OTH"],
        "peer_group": ["analog_power"] * 4 + ["other"],
        "ebitda_yield": [1 / 21.7, 1 / 33.1, 1 / 25.0, np.nan, 1 / 5.0],
        "earnings_yield": [1 / 27.9, -0.02, 1 / 30.0, 1 / 40.0, 0.5],
        "gross_margin": [0.59, 0.49, 0.55, 0.61, 0.10],
        "roic": [0.43, 0.50, 0.20, 0.10, 0.90],
        "net_debt_to_ebitda": [-0.3, np.inf, 1.0, 2.0, 0.0],
        "operating_margin_change": [0.143, -0.005, 0.02, 0.01, 0.0],
        "fcf_margin_change": [0.045, np.nan, 0.01, -0.01, 0.0],
    })


def quarterly_table(cik="0000000001", start_revenue=100.0):
    """Eight calendar quarters, FY2025 Q1 to FY2026 Q4, revenue rising 5 a quarter."""
    rows = []
    for i, (fy, q) in enumerate([(y, q) for y in (2025, 2026) for q in (1, 2, 3, 4)]):
        revenue = start_revenue + 5 * i
        end = pd.Timestamp(f"{fy}-{ENDS[q]}")
        lines = {"revenue": revenue, "gross_profit": revenue * 0.5, "operating_income": revenue * 0.2,
                 "operating_cash_flow": revenue * 0.3, "capex": revenue * 0.1}
        for metric, val in lines.items():
            rows.append({
                "cik10": cik, "metric": metric, "fiscal_year": fy, "quarter": q, "period_end": end, "val": val,
                "val_first": val, "method": "derived" if q == 4 else "reported", "source": "tag",
                "formula_derived": False, "known_on": end + pd.Timedelta(days=40), "val_ttm": val * 4,
            })
    return pd.DataFrame(rows)


# ---- peer medians --------------------------------------------------------

def test_peer_median_is_within_the_group_and_skips_blanks():
    m = metrics_table()
    assert peer_median(m, "analog_power", "gross_margin") == pytest.approx(0.57)  # 0.49 0.55 0.59 0.61
    assert peer_median(m, "analog_power", "ebitda_yield") == pytest.approx(1 / 25.0)  # NaN skipped, 3 values
    assert math.isinf(peer_median(m, "analog_power", "net_debt_to_ebitda")) is False
    assert peer_median(m, "analog_power", "net_debt_to_ebitda") == pytest.approx(1.5)  # inf counts as highest
    assert peer_median(None, "analog_power", "roic") is None
    assert peer_median(m, "analog_power", "not_a_metric") is None


def test_comparisons_carry_the_peer_median_in_display_form():
    scores, ranks, fundamentals = tables()
    facts = fact_sheet(pd.Series(pair_row()), scores, ranks, fundamentals, None, 12, metrics_table())
    by = {c["metric"]: c for c in facts["comparisons"]}
    assert by["gross_margin"]["peer_median"] == "57%"
    assert by["ebitda_yield"]["peer_median"] == "25.0x"  # median of the long's metric (EBITDA yield)
    assert facts["peer_group_size"] == 4
    text = render_template(facts, {"ticker": "AAA", "release": None, "annual": None},
                           {"ticker": "BBB", "release": None, "annual": None})
    assert "| Metric | AAA | BBB | Peer median | Stronger |" in text
    assert "| Gross margin | 59% | 49% | 57% | AAA |" in text
    assert "across the 4 companies in analog_power, both legs included" in text


# ---- quarterly trends ----------------------------------------------------

def test_quarter_trends_compute_growth_and_margins():
    trends = quarter_trends(quarterly_table(), "0000000001", pd.Timestamp("2027-03-01"))  # FY2026 Q4 filed Feb 2027
    assert [t["quarter"] for t in trends] == ["FY2026 Q1", "FY2026 Q2", "FY2026 Q3", "FY2026 Q4"]
    first = trends[0]  # revenue 120 vs 100 a year earlier
    assert first["revenue_growth"] == pytest.approx(0.2) and first["ended"] == "2026-03-31"
    assert (first["gross_margin"], first["operating_margin"]) == (0.5, 0.2)
    assert first["fcf_margin_ttm"] == pytest.approx(0.2)  # (0.3 - 0.1) of TTM revenue
    assert [t["derived"] for t in trends] == [False, False, False, True]


def test_quarter_trends_respect_as_of_and_missing_history():
    known = quarter_trends(quarterly_table(), "0000000001", pd.Timestamp("2026-06-01"))
    assert known[-1]["quarter"] == "FY2026 Q1"  # Q2 is filed in August
    early = quarter_trends(quarterly_table(), "0000000001", pd.Timestamp("2025-06-01"))
    assert [t["quarter"] for t in early] == ["FY2025 Q1"] and early[0]["revenue_growth"] is None
    assert quarter_trends(quarterly_table(), "0000000009", AS_OF) == []
    assert quarter_trends(None, "0000000001", AS_OF) == []


def test_trends_appear_in_the_fact_sheet_and_the_memo():
    scores, ranks, fundamentals = tables()
    fundamentals["cik10"] = ["0000000001", "0000000002", "0000000003"]
    quarterly = pd.concat([quarterly_table("0000000001"), quarterly_table("0000000002", 200.0)])
    deals = {"BBB": {"status": "closed", "evidence": "Completed the combination."}}
    facts = fact_sheet(pd.Series(pair_row()), scores, ranks, fundamentals, deals, 12, metrics_table(), quarterly)
    assert len(facts["companies"]["long"]["trends"]) == 4
    text = render_template(facts, {"ticker": "AAA", "release": None, "annual": None},
                           {"ticker": "BBB", "release": None, "annual": None})
    _, sections = split_sections(text)
    recent = sections["Recent quarters"]
    # As of 2026-11-11 the latest known quarter is FY2026 Q3; FY2025 Q4 is a derived quarter.
    assert "| FY2026 Q1 | 2026-03-31 | +20% | 50.0% | 20.0% | 20.0% |" in recent
    assert "| FY2025 Q4* | 2025-12-31 | n/a |" in recent and "FY2026 Q4" not in recent
    assert "Quarter derived from year-to-date figures" in recent
    assert "Recent deal: year-on-year growth is not like-for-like." in recent


# ---- what the model is allowed to see ------------------------------------

def test_filing_facts_leave_out_everything_that_moves_with_the_price():
    scores, ranks, fundamentals = tables()
    facts = fact_sheet(pd.Series(pair_row()), scores, ranks, fundamentals,
                       {"BBB": {"status": "closed", "evidence": "Completed it."}}, 12, metrics_table())
    material = filing_facts(facts)
    assert set(material) == {"pair_id", "long", "short", "peer_group", "companies", "fundamentals"}
    assert {c["pillar"] for c in material["fundamentals"]} == {"quality", "direction"}
    assert {"label": "ROIC", "pillar": "quality", "AAA": "43%", "BBB": "50%", "stronger": "BBB"} in material["fundamentals"]
    long = material["companies"]["long"]
    assert set(long) <= {"ticker", "latest_quarter", "quarter_ended", "deal_status", "deal_evidence", "data_notes", "trends"}
    assert material["companies"]["short"]["deal_evidence"] == "Completed it."
    text = str(material)
    for leaked in ("price", "market_cap", "composite", "hedge", "spread", "21.7x", "peer_median", "rank"):
        assert leaked not in text


# ---- what counts as a new filing -----------------------------------------

def filing(accn, form, filed, items=""):
    return {"cik10": "0000000001", "accn": accn, "form": form, "filed": pd.Timestamp(filed), "items": items}


def test_latest_relevant_filing_ignores_insider_and_other_forms():
    filings = pd.DataFrame([
        filing("Q-1", "10-Q", "2026-08-01"),
        filing("E-1", "8-K", "2026-10-20", "2.02,9.01"),
        filing("F-4", "4", "2026-11-01"),                    # insider trade
        filing("X-1", "8-K", "2026-11-02", "5.02"),          # officer change
        filing("S-1", "SC 13G", "2026-11-03"),
        filing("K-2", "10-K", "2027-02-20"),                 # after as_of
    ])
    assert latest_relevant_filing(filings, "1", AS_OF) == ("E-1", pd.Timestamp("2026-10-20"))
    amended = pd.concat([filings, pd.DataFrame([filing("QA-1", "10-Q/A", "2026-11-05")])])
    assert latest_relevant_filing(amended, "1", AS_OF)[0] == "QA-1"
    assert latest_relevant_filing(filings, "2", AS_OF) == (None, None)
