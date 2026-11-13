"""The web app: data shaping, charts, and every page rendering (no network)."""

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from screener.app import charts
from screener.app.data import (
    comparisons_view,
    deal_flag,
    group_view,
    load_manifest,
    load_table,
    memo_body,
    pair_label,
    pairs_view,
    quarterly_ratios,
    spread_frame,
)

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
ENDS = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}


def pair(long, short, rank, passes=True, group="g", reasons=""):
    return {"pair_id": f"{long}/{short}", "rank": rank if passes else pd.NA, "long": long, "short": short,
            "peer_group": group, "gap": 0.5, "gap_valuation": 0.4, "gap_quality": -0.1, "gap_direction": 0.6,
            "correlation": 0.63, "hedge_ratio": 1.12, "share_ratio": 0.74, "spread_z": -1.1, "spread_chg_3m": -0.04,
            "spread_chg_12m": -0.16, "passes": passes, "reasons": reasons, "notes": "", "thin_group": False,
            "deal_long": "none", "deal_short": "closed" if short == "BBB" else "none",
            "as_of": pd.Timestamp("2026-11-13")}


def pairs_table():
    frame = pd.DataFrame([pair("AAA", "BBB", 1), pair("AAA", "CCC", 2, group="h"),
                          pair("CCC", "BBB", None, passes=False, reasons="correlation 0.40 below 0.50")])
    frame["rank"] = frame["rank"].astype("Int64")
    return frame


def quarterly(cik="1"):
    rows = []
    for i, (fy, q) in enumerate([(y, q) for y in (2025, 2026) for q in (1, 2, 3, 4)]):
        revenue = 100.0 + 10 * i
        for metric, val in {"revenue": revenue, "gross_profit": revenue * 0.5, "operating_income": revenue * 0.2,
                            "operating_cash_flow": revenue * 0.3, "capex": revenue * 0.1}.items():
            rows.append({"cik10": cik, "metric": metric, "fiscal_year": fy, "quarter": q,
                         "period_end": pd.Timestamp(f"{fy}-{ENDS[q]}"), "val": val,
                         "method": "derived" if q == 4 else "reported",
                         "known_on": pd.Timestamp(f"{fy}-{ENDS[q]}") + pd.Timedelta(days=40), "val_ttm": val * 4})
    return pd.DataFrame(rows)


FACTS = {"long": "AAA", "short": "BBB", "comparisons": [
    {"pillar": "valuation", "label": "EV/EBITDA", "long": "21.7x", "short": "30.0x", "peer_median": "28.7x", "edge": 0.8},
    {"pillar": "quality", "label": "ROIC", "long": "43%", "short": "50%", "peer_median": None, "edge": -0.5},
]}


@pytest.fixture
def published(tmp_path, monkeypatch):
    folder = tmp_path / "published"
    folder.mkdir()
    dates = pd.bdate_range("2025-01-01", periods=60)
    tables = {
        "pairs": pairs_table(),
        "pair_spreads": pd.DataFrame({"pair_id": "AAA/BBB", "date": dates, "spread": [i / 1000 for i in range(60)]}),
        "deals": pd.DataFrame([{"ticker": "BBB", "status": "closed", "source": "detected", "filed": pd.Timestamp("2026-11-03"),
                                "evidence": "Completed the combination."}]),
        "scores": pd.DataFrame([
            {"ticker": t, "peer_group": g, "composite": c, "valuation": c, "quality": c, "direction": c,
             "rank_in_group": r, "group_size": 3, "thin_group": True}
            for t, g, c, r in (("AAA", "g", 0.8, 1), ("BBB", "g", 0.3, 2), ("CCC", "h", 0.5, 1))
        ] + [{"ticker": "MU", "peer_group": None, "composite": None, "valuation": None, "quality": None,
              "direction": None, "rank_in_group": None, "group_size": None, "thin_group": False}]),
        "metrics": pd.DataFrame([{"ticker": t, "peer_group": g, "ev_ebitda": 20.0, "ev_ebit": 25.0, "pe": 30.0,
                                  "fcf_yield": 0.03, "gross_margin": 0.5, "operating_margin": 0.2, "roic": 0.25,
                                  "revenue_growth": 0.1} for t, g in (("AAA", "g"), ("BBB", "g"), ("CCC", "h"))]),
        "fundamentals": pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "cik10": ["1", "2", "3"]}),
        "companies": pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "name": ["Aco", "Bco", "Cco"]}),
        "metrics_quarterly": pd.concat([quarterly("1"), quarterly("2")]),
        "memos": pd.DataFrame([
            {"pair_id": "AAA/BBB", "memo": "# Long AAA / Short BBB\nheader line\n\n## Why this pair\nTemplate text.",
             "memo_ai": "# Long AAA / Short BBB\n\n## Analyst view\nAI text.", "ai_status": "cached", "ai_note": "",
             "ai_model": "claude-haiku-5-5", "ai_written": "2026-11-05", "fact_sheet": json.dumps(FACTS)},
            {"pair_id": "AAA/CCC", "memo": "# Long AAA / Short CCC\n\nTemplate only.", "memo_ai": None,
             "ai_status": "off", "ai_note": "AI memos off (no ANTHROPIC_API_KEY in .env).", "ai_model": None,
             "ai_written": None, "fact_sheet": json.dumps({**FACTS, "short": "CCC"})},
        ]),
        "validation": pd.DataFrame([{"check": "negative_value", "severity": "error", "ticker": "AAA", "metric": "x",
                                     "fiscal_year": 2025, "quarter": 4, "value": -1.0, "expected": None, "detail": "d"}]),
        "model_runs": pd.DataFrame([{"run_at": "2026-11-13", "api_calls": 2, "estimated_cost_usd": 0.0012}]),
        "pipeline_runs": pd.DataFrame([{"started_at": "2026-11-13 23:00:00", "status": "ok", "steps": "all ok"}]),
    }
    for name, frame in tables.items():
        frame.to_parquet(folder / f"{name}.parquet", index=False)
    (folder / "manifest.json").write_text(json.dumps({"as_of": "2026-11-13", "published_at": "2026-11-13 23:10:00 UTC",
                                                      "passing_pairs": 2}))
    monkeypatch.setenv("SCREENER_PUBLISHED_DIR", str(folder))
    return folder


# ---- data shaping --------------------------------------------------------

def test_pairs_view_filters_and_scales_percentages():
    view = pairs_view(pairs_table())
    assert view["pair_id"].tolist() == ["AAA/BBB", "AAA/CCC"]
    first = view.iloc[0]
    assert (first["Spread 12m %"], first["Spread 3m %"]) == pytest.approx((-16.0, -4.0))
    assert first["Deals"] == "BBB: recent deal" and first["Status"] == "passes"
    assert pairs_view(pairs_table(), groups=["h"])["pair_id"].tolist() == ["AAA/CCC"]
    everything = pairs_view(pairs_table(), include_failing=True)
    assert everything["pair_id"].tolist()[-1] == "CCC/BBB"
    assert everything["Status"].tolist()[-1] == "fails: correlation 0.40 below 0.50"


def test_labels_and_flags():
    row = pairs_table().iloc[0]
    assert pair_label(row) == "1. Long AAA / Short BBB (g)"
    assert deal_flag("pending_target", "none", "AAA", "BBB") == "AAA: being acquired"
    assert deal_flag("none", "none", "AAA", "BBB") == ""


def test_spread_frame():
    spreads = pd.DataFrame({"pair_id": ["X/Y", "A/B", "A/B"], "date": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-02"]),
                            "spread": [0.5, 0.02, 0.01]})
    frame = spread_frame(spreads, "A/B")
    assert frame["date"].tolist() == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")]
    assert frame["spread_pct"].tolist() == pytest.approx([1.0, 2.0])


def test_quarterly_ratios():
    frame = quarterly_ratios(quarterly(), "1", "AAA", count=4)
    assert set(frame["line"]) == {"Revenue growth vs year-ago quarter", "Gross margin", "Operating margin",
                                  "Free cash flow margin (TTM)"}
    growth = frame[frame["line"] == "Revenue growth vs year-ago quarter"]
    assert growth["quarter"].tolist() == ["FY2026 Q1", "FY2026 Q2", "FY2026 Q3", "FY2026 Q4"]
    assert growth["value"].iloc[0] == pytest.approx(140 / 100 - 1)
    assert growth["derived"].tolist() == [False, False, False, True]
    margins = frame[frame["line"] == "Gross margin"]["value"]
    assert (margins == 0.5).all()
    # With the full history, the first year has margins but no growth.
    all_quarters = quarterly_ratios(quarterly(), "1", "AAA", count=12)
    assert len(all_quarters[all_quarters["line"] == "Gross margin"]) == 8
    assert len(all_quarters[all_quarters["line"] == "Revenue growth vs year-ago quarter"]) == 4
    assert quarterly_ratios(quarterly(), "9", "ZZZ").empty


def test_group_view(published):
    t = {name: load_table(published, name) for name in ("scores", "metrics", "companies", "deals")}
    view = group_view(t["scores"], t["metrics"], t["companies"], t["deals"], "g")
    assert view["Ticker"].tolist() == ["AAA", "BBB"] and view["Company"].tolist() == ["Aco", "Bco"]
    assert view["Deal"].tolist() == ["", "recent deal"]
    assert view["Gross margin %"].tolist() == [50.0, 50.0]


def test_memo_helpers():
    assert memo_body("# Title\nheader\n\n## A") == "header\n\n## A"
    assert memo_body("no title") == "no title"
    view = comparisons_view(json.dumps(FACTS))
    assert view.columns.tolist() == ["Pillar", "Metric", "AAA", "BBB", "Peer median", "Stronger"]
    assert view["Stronger"].tolist() == ["AAA", "BBB"] and view["Peer median"].tolist() == ["28.7x", "n/a"]


def test_loaders_tolerate_missing_files(tmp_path):
    assert load_table(tmp_path, "pairs") is None and load_manifest(tmp_path) is None


# ---- charts --------------------------------------------------------------

def test_charts_use_the_role_colors_in_each_mode():
    frame = spread_frame(pd.DataFrame({"pair_id": "A/B", "date": pd.bdate_range("2026-01-01", periods=5),
                                       "spread": [0.0, 0.01, 0.02, 0.01, 0.0]}), "A/B")
    for mode in ("light", "dark"):
        spec = charts.spread_chart(frame, "A", "B", mode).to_dict()
        assert charts.PALETTE[mode]["long"] in json.dumps(spec)
        assert spec["config"]["axis"]["gridDash"] == []
    trends = pd.concat([quarterly_ratios(quarterly("1"), "1", "AAA"), quarterly_ratios(quarterly("2"), "2", "BBB")])
    spec = charts.trend_chart(trends, "Gross margin", "AAA", "BBB", "dark").to_dict()
    text = json.dumps(spec)
    assert '"Long AAA"' in text and '"Short BBB"' in text
    assert charts.PALETTE["dark"]["long"] in text and charts.PALETTE["dark"]["short"] in text


# ---- every page renders --------------------------------------------------

@pytest.mark.parametrize("page, header", [
    ("Pairs", "Ranked pairs"), ("Pair detail", "Long AAA / Short BBB"),
    ("Peer groups", "Peer groups"), ("Data status", "Data status"),
])
def test_page_renders(published, page, header):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["page"] = page
    at.run()
    assert not at.exception
    assert [h.value for h in at.header] == [header]


def test_pair_detail_shows_ai_memo_or_why_not(published):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["page"] = "Pair detail"
    at.session_state["pair_id"] = "AAA/CCC"
    at.run()
    assert not at.exception
    assert any("No AI memo for this pair. AI memos off" in i.value for i in at.info)


def test_without_published_data_the_app_says_what_to_run(tmp_path, monkeypatch):
    monkeypatch.setenv("SCREENER_PUBLISHED_DIR", str(tmp_path / "empty"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert "python -m screener.pipeline --publish" in at.error[0].value
