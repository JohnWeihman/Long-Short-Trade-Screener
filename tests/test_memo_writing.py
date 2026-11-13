import json
import math

import numpy as np
import pandas as pd
import pytest

from screener.memos.build import MEMO_COLUMNS, build_memos, write_markdown
from screener.memos.facts import fact_sheet, show
from screener.memos.sources import MemoConfig
from screener.memos.write import _excerpt, company_sources, inputs_hash, render_template

AS_OF = pd.Timestamp("2026-11-11")


def pair_row(long="AAA", short="BBB", rank=1, passes=True, **over):
    row = {
        "pair_id": f"{long}/{short}", "peer_group": "analog_power", "long": long, "short": short, "gap": 0.44,
        "gap_valuation": 0.40, "gap_quality": 0.25, "gap_direction": 0.67, "thin_group": False,
        "correlation": 0.63, "overlap_weeks": 104, "beta_long": 1.16, "beta_short": 1.03, "hedge_ratio": 1.1246,
        "share_ratio": 0.74, "spread_z": -1.1, "spread_chg_1m": -0.12, "spread_chg_3m": -0.04, "spread_chg_12m": -0.16,
        "passes": passes, "rank": rank if passes else pd.NA, "as_of": AS_OF, "notes": "", "reasons": "",
    }
    row.update(over)
    return row


def tables():
    scores = pd.DataFrame([
        {"ticker": "AAA", "composite": 0.79, "rank_in_group": 1, "group_size": 6, "valuation": 0.75, "quality": 0.70, "direction": 0.93},
        {"ticker": "BBB", "composite": 0.36, "rank_in_group": 6, "group_size": 6, "valuation": 0.35, "quality": 0.45, "direction": 0.27},
        {"ticker": "CCC", "composite": 0.50, "rank_in_group": 3, "group_size": 6, "valuation": 0.50, "quality": np.nan, "direction": 0.50},
    ])
    def rank(ticker, pillar, metric, value, r, used=None):
        return {"ticker": ticker, "pillar": pillar, "metric": metric, "used_metric": used or metric, "value": value, "rank": r}
    ranks = pd.DataFrame([
        rank("AAA", "valuation", "ebitda_yield", 1 / 21.7, 0.8), rank("BBB", "valuation", "ebitda_yield", 1 / 33.1, 0.2, "ebit_yield"),
        rank("AAA", "valuation", "earnings_yield", 1 / 27.9, 0.6), rank("BBB", "valuation", "earnings_yield", -0.02, 0.0),
        rank("AAA", "quality", "gross_margin", 0.59, 1.0), rank("BBB", "quality", "gross_margin", 0.49, 0.4),
        rank("AAA", "quality", "net_debt_to_ebitda", -0.3, 0.8), rank("BBB", "quality", "net_debt_to_ebitda", np.inf, 0.0),
        rank("AAA", "quality", "roic", 0.43, 0.4), rank("BBB", "quality", "roic", 0.50, 0.9),
        rank("AAA", "direction", "operating_margin_change", 0.143, 1.0), rank("BBB", "direction", "operating_margin_change", -0.005, 0.0),
        rank("AAA", "direction", "fcf_margin_change", 0.045, 0.6), rank("BBB", "direction", "fcf_margin_change", np.nan, np.nan),
    ])
    def fund(ticker, price, cap, notes=""):
        return {"ticker": ticker, "fiscal_year": 2026, "quarter": 3, "period_end": pd.Timestamp("2026-09-27"),
                "price": price, "market_cap": cap, "revenue_ttm": 5.0e9, "notes": notes}
    fundamentals = pd.DataFrame([fund("AAA", 245.40, 38.3e9, "long_term_debt not reported: taken as 0"),
                                 fund("BBB", 372.53, 295.6e9, "no D&A: EBITDA unavailable; cash not reported: taken as 0"),
                                 fund("CCC", 10.0, 1e9)])
    return scores, ranks, fundamentals


def text_tables():
    def section(ticker, kind, accn, filed, name, text, ok=True):
        return {"cik10": "1", "ticker": ticker, "kind": kind, "accn": accn, "filed": pd.Timestamp(filed),
                "section": name, "text": text, "chars": len(text), "ok": ok, "problem": ""}
    sections = pd.DataFrame([
        section("AAA", "earnings_release", "A-E1", "2026-07-28", "headline", "AAA Reports Second Quarter Results"),
        section("AAA", "earnings_release", "A-E1", "2026-07-28", "outlook", "Old guidance."),
        section("AAA", "earnings_release", "A-E2", "2026-10-28", "headline", "AAA Reports Third Quarter 2026 Results"),
        section("AAA", "earnings_release", "A-E2", "2026-10-28", "outlook",
                "Guidance for the fourth quarter of 2026 is revenue of $1,250 million to $1,350 million."),
        section("AAA", "annual_report", "A-K", "2026-02-19", "risk_headings",
                "Our business is cyclical.\nWe depend on a few customers.\nWe operate internationally.\nA fourth risk."),
        section("BBB", "earnings_release", "B-E2", "2026-08-13", "headline", "BBB ANNOUNCES THIRD QUARTER 2026 RESULTS"),
        section("BBB", "earnings_release", "B-E2", "2026-08-13", "outlook", "", ok=False),
    ])
    def doc(ticker, kind, accn, filed):
        return {"ticker": ticker, "kind": kind, "accn": accn, "filed": pd.Timestamp(filed), "url": f"https://sec.example/{accn}"}
    documents = pd.DataFrame([
        doc("AAA", "earnings_release", "A-E1", "2026-07-28"), doc("AAA", "earnings_release", "A-E2", "2026-10-28"),
        doc("AAA", "annual_report", "A-K", "2026-02-19"), doc("BBB", "earnings_release", "B-E2", "2026-08-13"),
    ])
    return sections, documents


def facts_for(pair=None, deals=None):
    scores, ranks, fundamentals = tables()
    return fact_sheet(pd.Series(pair or pair_row()), scores, ranks, fundamentals, deals, 12)


# ---- showing values ----------------------------------------------------

@pytest.mark.parametrize("metric, value, text", [
    ("ebitda_yield", 1 / 21.7, "21.7x"), ("earnings_yield", -0.02, "negative earnings"),
    ("fcf_yield", 0.027, "2.7%"), ("gross_margin", 0.594, "59%"),
    ("operating_margin_change", 0.143, "+14.3 pts"), ("fcf_margin_change", -0.021, "-2.1 pts"),
    ("net_debt_to_ebitda", 1.25, "1.2x"), ("net_debt_to_ebitda", -0.3, "net cash (-0.3x)"),
    ("net_debt_to_ebit", math.inf, "debt with no earnings"), ("roic", float("nan"), "n/a"), ("roic", None, "n/a"),
])
def test_show(metric, value, text):
    assert show(metric, value) == text


# ---- fact sheet --------------------------------------------------------

def test_fact_sheet_is_plain_data_with_the_headline_numbers():
    facts = facts_for()
    json.dumps(facts)  # must serialize as-is
    assert (facts["pair_id"], facts["as_of"], facts["rank"], facts["passing_pairs"]) == ("AAA/BBB", "2026-11-11", 1, 12)
    assert facts["composite_gap"] == 0.44 and facts["pillar_gaps"] == {"valuation": 0.4, "quality": 0.25, "direction": 0.67}
    long = facts["companies"]["long"]
    assert (long["composite"], long["rank_in_group"], long["group_size"]) == (0.79, 1, 6)
    assert (long["latest_quarter"], long["quarter_ended"], long["price"], long["market_cap_bn"]) == ("FY2026 Q3", "2026-09-27", 245.4, 38.3)
    assert facts["companies"]["short"]["data_notes"] == "no D&A: EBITDA unavailable"  # routine note dropped
    assert long["data_notes"] == ""
    assert facts["trade"]["hedge_ratio"] == 1.12 and facts["trade"]["spread_chg_12m"] == -0.16


def test_comparisons_use_the_metric_each_company_was_ranked_on():
    by = {c["metric"]: c for c in facts_for()["comparisons"]}
    assert by["ebitda_yield"]["label"] == "EV/EBITDA / EV/EBIT"  # short fell back to EBIT
    assert (by["ebitda_yield"]["long"], by["ebitda_yield"]["short"]) == ("21.7x", "33.1x")
    assert (by["earnings_yield"]["long"], by["earnings_yield"]["short"]) == ("27.9x", "negative earnings")
    assert by["net_debt_to_ebitda"]["short"] == "debt with no earnings"
    assert "fcf_margin_change" not in by  # not ranked for the short


def test_drivers_for_and_against_the_trade():
    facts = facts_for()
    assert [c["metric"] for c in facts["for_the_trade"]] == [
        "operating_margin_change", "net_debt_to_ebitda", "ebitda_yield", "earnings_yield"]  # widest edge first, top 4
    assert [c["metric"] for c in facts["against_the_trade"]] == ["roic"]
    assert facts["against_the_trade"][0]["edge"] == -0.5


def test_deal_flags_and_missing_values_in_the_fact_sheet():
    facts = facts_for(deals={"BBB": {"status": "closed", "evidence": "Completed the Qorvo combination."},
                             "AAA": {"status": "none", "evidence": "ignored"}})
    assert facts["companies"]["short"]["deal_status"] == "closed"
    assert facts["companies"]["short"]["deal_evidence"] == "Completed the Qorvo combination."
    assert facts["companies"]["long"]["deal_evidence"] == ""
    partial = facts_for(pair_row(short="CCC", spread_z=np.nan, spread_chg_12m=np.nan))
    assert partial["companies"]["short"]["pillars"]["quality"] is None
    assert partial["trade"]["spread_z"] is None and partial["trade"]["spread_chg_12m"] is None


# ---- sources -----------------------------------------------------------

def test_company_sources_take_the_latest_release_and_first_risk_headings():
    sections, documents = text_tables()
    got = company_sources("AAA", sections, documents, risk_count=3)
    assert got["release"]["headline"] == "AAA Reports Third Quarter 2026 Results"
    assert got["release"]["accn"] == "A-E2" and got["release"]["filed"] == "2026-10-28"
    assert got["annual"]["risk_headings"] == ["Our business is cyclical.", "We depend on a few customers.", "We operate internationally."]
    bbb = company_sources("BBB", sections, documents)
    assert bbb["release"]["outlook"] == "" and bbb["annual"] is None  # failed outlook is not used
    assert company_sources("ZZZ", sections, documents) == {"ticker": "ZZZ", "release": None, "annual": None}


def test_excerpt_cuts_at_a_sentence_or_word():
    text = "First sentence here. " * 30
    cut = _excerpt(text, 100)
    assert cut.endswith("here. ...") and len(cut) <= 104
    assert _excerpt("short text") == "short text"
    assert _excerpt("word " * 200, 50).endswith(" ...") and "  " not in _excerpt("a  b\n c")


# ---- the memo ----------------------------------------------------------

def memo(pair=None, deals=None):
    sections, documents = text_tables()
    facts = facts_for(pair, deals)
    return render_template(facts, company_sources(facts["long"], sections, documents),
                           company_sources(facts["short"], sections, documents))


def test_memo_structure_and_numbers_come_from_the_fact_sheet():
    text = memo()
    assert text.startswith("# Long AAA / Short BBB\nanalog_power | as of 2026-11-11 | Rank 1 of 12 passing pairs")
    for heading in ("## Why this pair", "## What each company said", "## Risks and caveats", "## Trade construction", "## Sources"):
        assert heading in text
    assert "**Composite score 0.79 vs 0.36 (gap 0.44).**" in text
    assert "| Valuation | 0.75 | 0.35 | AAA ahead |" in text
    assert "- Operating margin change vs a year ago: +14.3 pts vs -0.5 pts" in text
    assert "What favors BBB:" in text and "- ROIC: 43% vs 50%" in text
    assert "short $1.12 of BBB per $1.00 of AAA, or 0.74 BBB shares per AAA share at $245.40 and $372.53" in text


def test_memo_quotes_filings_and_links_sources():
    text = memo()
    assert "**AAA** (2026-10-28): AAA Reports Third Quarter 2026 Results" in text
    assert "> Guidance for the fourth quarter of 2026 is revenue of $1,250 million to $1,350 million." in text
    assert "Old guidance." not in text
    assert "No outlook or guidance passage was found in the release." in text  # BBB
    assert "Risks AAA lists first in its 10-K:" in text and "- We depend on a few customers." in text
    assert "A fourth risk." not in text
    assert "- AAA earnings release, filed 2026-10-28: [A-E2](https://sec.example/A-E2)" in text
    assert "- AAA 10-K, filed 2026-02-19: [A-K](https://sec.example/A-K)" in text


def test_memo_states_caveats():
    text = memo(pair_row(thin_group=True), {"BBB": {"status": "closed", "evidence": "Completed the Qorvo combination."}})
    assert "- **BBB recently completed a deal, so its results may not be comparable with earlier periods.** Completed the Qorvo combination." in text
    assert "- The peer group is small, so ranks within it are coarse." in text
    assert "AAA has underperformed the hedged BBB position by 16%" in text and "z-score against the last year is -1.1" in text
    assert "- Data note, BBB: no D&A: EBITDA unavailable." in text
    assert "outperformed" in memo(pair_row(spread_chg_12m=0.34))


def test_memo_for_a_company_with_no_filing_text():
    scores, ranks, fundamentals = tables()
    facts = fact_sheet(pd.Series(pair_row()), scores, ranks, fundamentals, None, 12)
    empty = {"ticker": "BBB", "release": None, "annual": None}
    sections, documents = text_tables()
    text = render_template(facts, company_sources("AAA", sections, documents), empty)
    assert "**BBB**: no earnings release could be read." in text


def test_inputs_hash_changes_with_inputs():
    sections, documents = text_tables()
    facts = facts_for()
    a, b = company_sources("AAA", sections, documents), company_sources("BBB", sections, documents)
    first = inputs_hash(facts, a, b)
    assert first == inputs_hash(facts_for(), a, b) and len(first) == 16
    assert first != inputs_hash(facts_for(pair_row(gap=0.45)), a, b)
    assert first != inputs_hash(facts, {**a, "release": None}, b)


# ---- table of memos ----------------------------------------------------

def test_build_memos_covers_top_passing_pairs_only(cache):
    scores, ranks, fundamentals = tables()
    sections, documents = text_tables()
    pairs = pd.DataFrame([
        pair_row("AAA", "BBB", rank=2), pair_row("AAA", "CCC", rank=1), pair_row("CCC", "BBB", passes=False),
    ])
    deals = pd.DataFrame([{"ticker": "CCC", "status": "pending_other", "evidence": "CCC agreed to acquire X."}])
    memos = build_memos(pairs, scores, ranks, fundamentals, deals, sections, documents, MemoConfig(top_pairs=5))
    assert list(memos.columns) == MEMO_COLUMNS
    assert memos[["rank", "pair_id", "writer"]].values.tolist() == [[1, "AAA/CCC", "template"], [2, "AAA/BBB", "template"]]
    assert "Rank 1 of 2 passing pairs" in memos.loc[0, "memo"] and "CCC is party to a pending deal" in memos.loc[0, "memo"]
    assert json.loads(memos.loc[1, "fact_sheet"])["pair_id"] == "AAA/BBB"
    assert json.loads(memos.loc[1, "sources"])[0]["release"]["accn"] == "A-E2"
    assert len(build_memos(pairs, scores, ranks, fundamentals, None, sections, documents, MemoConfig(top_pairs=1))) == 1

    folder = write_markdown(cache, memos)
    assert sorted(p.name for p in folder.glob("*.md")) == ["01_AAA_CCC.md", "02_AAA_BBB.md"]
    assert (folder / "02_AAA_BBB.md").read_text(encoding="utf-8") == memos.loc[1, "memo"]
    write_markdown(cache, memos.head(1))  # stale files are removed
    assert [p.name for p in folder.glob("*.md")] == ["01_AAA_CCC.md"]
