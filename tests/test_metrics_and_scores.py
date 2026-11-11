import numpy as np
import pandas as pd
import pytest

from screener.scoring.metrics import COLUMNS, RANKED, compute_metrics, leverage, over_positive
from screener.scoring.report import company_detail, group_table
from screener.scoring.scores import (
    MetricSpec,
    ScoringConfig,
    ScoringConfigError,
    load_scoring_config,
    parse_scoring_config,
    percentile_rank,
    rank_metrics,
    score,
)

AS_OF = pd.Timestamp("2026-02-15")


def fund(ticker, group="g", **overrides):
    """A healthy company: revenue 1,000, 20% operating margin, EV 5,000."""
    row = {
        "cik10": ticker.rjust(10, "0"), "ticker": ticker, "peer_group": group, "as_of": AS_OF,
        "fiscal_year": 2025, "quarter": 4, "period_end": pd.Timestamp("2025-12-31"),
        "revenue_ttm": 1000.0, "revenue_ttm_1y_ago": 800.0, "revenue_ttm_2y_ago": 700.0,
        "gross_profit_ttm": 500.0, "operating_income_ttm": 200.0, "operating_income_ttm_1y_ago": 120.0,
        "ebitda_ttm": 250.0, "net_income_ttm": 150.0, "fcf_ttm": 100.0, "fcf_ttm_1y_ago": 60.0,
        "market_cap": 4800.0, "enterprise_value": 5000.0, "net_debt": 200.0, "roic": 0.25,
    }
    row.update(overrides)
    return row


def metrics_for(*rows):
    return compute_metrics(pd.DataFrame(rows))


# ---- ratios ------------------------------------------------------------

def test_healthy_company_ratios():
    m = metrics_for(fund("A")).iloc[0]
    assert list(metrics_for(fund("A")).columns) == COLUMNS
    assert m["ebitda_yield"] == pytest.approx(0.05) and m["ev_ebitda"] == pytest.approx(20.0)
    assert m["ebit_yield"] == pytest.approx(0.04) and m["ev_ebit"] == pytest.approx(25.0)
    assert m["earnings_yield"] == pytest.approx(150 / 4800) and m["pe"] == pytest.approx(32.0)
    assert m["fcf_yield"] == pytest.approx(100 / 4800)
    assert m["sales_yield"] == pytest.approx(0.2) and m["ev_sales"] == pytest.approx(5.0)
    assert (m["gross_margin"], m["operating_margin"], m["roic"]) == (0.5, 0.2, 0.25)
    assert m["net_debt_to_ebitda"] == pytest.approx(0.8) and m["net_debt_to_ebit"] == pytest.approx(1.0)


def test_direction_metrics():
    m = metrics_for(fund("A")).iloc[0]
    assert m["operating_margin_change"] == pytest.approx(0.20 - 0.15)
    assert m["revenue_growth"] == pytest.approx(0.25)
    assert m["revenue_growth_1y_ago"] == pytest.approx(800 / 700 - 1)
    assert m["revenue_growth_acceleration"] == pytest.approx(0.25 - (800 / 700 - 1))
    assert m["fcf_margin_change"] == pytest.approx(0.10 - 0.075)


def test_loss_making_company_has_negative_yields_and_no_multiples():
    m = metrics_for(fund("A", operating_income_ttm=-100.0, ebitda_ttm=-50.0, net_income_ttm=-120.0, fcf_ttm=-80.0)).iloc[0]
    assert m["ebitda_yield"] == pytest.approx(-0.01) and m["earnings_yield"] == pytest.approx(-0.025)
    assert m["fcf_yield"] < 0 and m["operating_margin"] == pytest.approx(-0.1)
    assert np.isnan(m["ev_ebitda"]) and np.isnan(m["pe"]) and np.isnan(m["ev_ebit"])


def test_missing_ebitda_leaves_the_ebit_versions():
    m = metrics_for(fund("A", ebitda_ttm=np.nan)).iloc[0]
    assert np.isnan(m["ebitda_yield"]) and np.isnan(m["net_debt_to_ebitda"]) and np.isnan(m["ev_ebitda"])
    assert m["ebit_yield"] == pytest.approx(0.04) and m["net_debt_to_ebit"] == pytest.approx(1.0)


def test_non_positive_enterprise_value_gives_no_ev_ratios():
    m = metrics_for(fund("A", enterprise_value=-10.0)).iloc[0]
    assert np.isnan(m["ebitda_yield"]) and np.isnan(m["sales_yield"]) and np.isnan(m["ev_sales"])
    assert not np.isnan(m["earnings_yield"])  # market cap is still positive


def test_leverage_cases():
    debt = pd.Series([200.0, -300.0, 200.0, -300.0, 200.0])
    earnings = pd.Series([250.0, 250.0, -10.0, -10.0, np.nan])
    got = leverage(debt, earnings)
    assert got.iloc[0] == pytest.approx(0.8)       # ordinary
    assert got.iloc[1] == pytest.approx(-1.2)      # net cash: negative, ranks best
    assert np.isinf(got.iloc[2])                   # debt and no earnings: worst
    assert np.isnan(got.iloc[3])                   # net cash and losses: not meaningful
    assert np.isnan(got.iloc[4])                   # earnings unknown


def test_over_positive():
    got = over_positive(pd.Series([1.0, 1.0, 1.0, np.nan]), pd.Series([2.0, 0.0, -2.0, 2.0]))
    assert got.iloc[0] == 0.5 and got.iloc[1:].isna().all()


def test_company_without_fundamentals_gets_blank_metrics():
    frame = pd.DataFrame([fund("A"), {"cik10": None, "ticker": "GONE", "peer_group": "g", "as_of": AS_OF}])
    m = compute_metrics(frame)
    assert m.loc[1, RANKED].isna().all() and m.loc[0, "gross_margin"] == 0.5


# ---- config ------------------------------------------------------------

def test_real_scoring_config_loads():
    cfg = load_scoring_config()
    assert list(cfg.pillars) == ["valuation", "quality", "direction"]
    assert cfg.pillar_weights == {"valuation": 1.0, "quality": 1.0, "direction": 1.0}
    assert cfg.pillars["valuation"][0] == MetricSpec("ebitda_yield", False, "ebit_yield")
    assert cfg.pillars["quality"][3] == MetricSpec("net_debt_to_ebitda", True, "net_debt_to_ebit")
    assert (cfg.min_group_size, cfg.min_metrics_per_pillar, cfg.min_pillars) == (4, 2, 2)


@pytest.mark.parametrize("raw, message", [
    ({}, "pillars"),
    ({"pillars": {"v": [{"metric": "magic"}]}}, "not a metric"),
    ({"pillars": {"v": [{"metric": "roic", "fallback": "magic"}]}}, "not a metric"),
    ({"pillars": {"v": [{"metric": "roic", "weight": 2}]}}, "unknown keys"),
    ({"pillars": {"v": [{"metric": "roic"}], "q": [{"metric": "roic"}]}}, "twice"),
    ({"pillars": {"v": []}}, "no metrics"),
    ({"pillars": {"v": [{"metric": "roic"}]}, "pillar_weights": {"v": 0}}, "pillar_weights"),
])
def test_bad_scoring_config_is_rejected(raw, message):
    with pytest.raises(ScoringConfigError, match=message):
        parse_scoring_config(raw)


# ---- ranking -----------------------------------------------------------

def test_percentile_rank_basics():
    ranks = percentile_rank(pd.Series([10.0, 30.0, 20.0, np.nan]))
    assert ranks.iloc[:3].tolist() == [0.0, 1.0, 0.5] and np.isnan(ranks.iloc[3])
    assert percentile_rank(pd.Series([10.0, 30.0, 20.0]), lower_is_better=True).tolist() == [1.0, 0.0, 0.5]


def test_percentile_rank_ties_infinity_and_too_few_values():
    assert percentile_rank(pd.Series([1.0, 1.0, 2.0])).tolist() == [0.25, 0.25, 1.0]
    worst = percentile_rank(pd.Series([0.5, np.inf, -1.0]), lower_is_better=True)
    assert worst.tolist() == [0.5, 0.0, 1.0]
    assert percentile_rank(pd.Series([5.0, np.nan])).isna().all()
    assert percentile_rank(pd.Series([], dtype=float)).empty


CFG = ScoringConfig(
    pillars={
        "valuation": (MetricSpec("ebitda_yield", fallback="ebit_yield"), MetricSpec("earnings_yield")),
        "quality": (MetricSpec("operating_margin"), MetricSpec("net_debt_to_ebitda", lower_is_better=True)),
        "direction": (MetricSpec("operating_margin_change"), MetricSpec("fcf_margin_change")),
    },
    pillar_weights={"valuation": 1.0, "quality": 1.0, "direction": 1.0},
    min_group_size=4, min_metrics_per_pillar=2, min_pillars=2,
)


def group():
    """Four companies: A is cheap, D is expensive; quality runs the other way."""
    return metrics_for(
        fund("A", enterprise_value=2500.0, market_cap=2400.0, operating_income_ttm=100.0, net_debt=600.0),
        fund("B", enterprise_value=4000.0, market_cap=3800.0, operating_income_ttm=150.0, net_debt=400.0),
        fund("C", enterprise_value=6000.0, market_cap=5800.0, operating_income_ttm=220.0, net_debt=100.0),
        fund("D", enterprise_value=9000.0, market_cap=9200.0, operating_income_ttm=300.0, net_debt=-200.0),
        fund("X", group=None),
    )


def test_ranks_are_within_group_and_oriented_best_to_one():
    ranks = rank_metrics(group(), CFG)
    assert "X" not in set(ranks["ticker"])  # unplaced: not ranked
    cheap = ranks[ranks["metric"] == "ebitda_yield"].set_index("ticker")["rank"]
    assert cheap.to_dict() == pytest.approx({"A": 1.0, "B": 2 / 3, "C": 1 / 3, "D": 0.0})
    debt = ranks[ranks["metric"] == "net_debt_to_ebitda"].set_index("ticker")["rank"]
    assert debt.to_dict() == pytest.approx({"A": 0.0, "B": 1 / 3, "C": 2 / 3, "D": 1.0})  # net cash is best


def test_ranks_do_not_cross_groups():
    two = metrics_for(fund("A", market_cap=1000.0), fund("B", market_cap=2000.0),
                      fund("C", group="h", market_cap=100.0), fund("D", group="h", market_cap=200.0))
    ranks = rank_metrics(two, CFG)
    earnings = ranks[ranks["metric"] == "earnings_yield"].set_index("ticker")["rank"]
    assert earnings.to_dict() == {"A": 1.0, "B": 0.0, "C": 1.0, "D": 0.0}


def test_fallback_metric_fills_the_slot_for_companies_missing_the_main_one():
    frame = group()
    frame.loc[frame["ticker"] == "B", "ebitda_yield"] = np.nan
    ranks = rank_metrics(frame, CFG)
    slot = ranks[ranks["metric"] == "ebitda_yield"].set_index("ticker")
    assert slot.loc["B", "used_metric"] == "ebit_yield"
    assert slot.loc["B", "value"] == pytest.approx(150 / 4000)
    # B's rank comes from ranking EBIT yield across all four companies.
    ebit_yield = frame.set_index("ticker")["ebit_yield"].drop("X")
    assert slot.loc["B", "rank"] == pytest.approx(percentile_rank(ebit_yield)["B"])
    # The others keep their EBITDA-yield rank among the three that have it.
    assert slot.loc["A", "used_metric"] == "ebitda_yield" and slot.loc["A", "rank"] == 1.0


def test_pillar_scores_composite_and_group_rank():
    frame = group()
    ranks = rank_metrics(frame, CFG)
    s = score(ranks, frame, CFG).set_index("ticker")
    assert s.loc["A", "valuation"] == pytest.approx(1.0) and s.loc["D", "valuation"] == pytest.approx(0.0)
    assert s.loc["D", "quality"] == pytest.approx(1.0) and s.loc["A", "quality"] == pytest.approx(0.0)
    assert s.loc["A", "composite"] == pytest.approx((s.loc["A", ["valuation", "quality", "direction"]]).mean())
    assert sorted(s.loc[["A", "B", "C", "D"], "rank_in_group"].tolist()) == [1, 2, 3, 4]
    assert s.loc["A", "group_size"] == 4 and not s.loc["A", "thin_group"]
    # The unplaced company is in the table but has no scores.
    assert pd.isna(s.loc["X", "composite"]) and pd.isna(s.loc["X", "rank_in_group"]) and s.loc["X", "pillars_scored"] == 0


def test_pillar_needs_enough_metrics_and_composite_enough_pillars():
    frame = group()
    frame.loc[frame["ticker"] == "C", ["operating_margin_change", "fcf_margin_change", "operating_margin"]] = np.nan
    s = score(rank_metrics(frame, CFG), frame, CFG).set_index("ticker")
    assert pd.isna(s.loc["C", "direction"]) and s.loc["C", "direction_metrics"] == 0
    assert pd.isna(s.loc["C", "quality"]) and s.loc["C", "quality_metrics"] == 1  # one metric is not enough
    assert s.loc["C", "pillars_scored"] == 1 and pd.isna(s.loc["C", "composite"])  # one pillar is not enough


def test_pillar_weights_change_the_composite():
    frame = group()
    ranks = rank_metrics(frame, CFG)
    value_only = ScoringConfig(CFG.pillars, {"valuation": 1.0, "quality": 0.0, "direction": 0.0}, 4, 2, 1)
    s = score(ranks, frame, value_only).set_index("ticker")
    assert s.loc["A", "composite"] == pytest.approx(s.loc["A", "valuation"])
    assert s.loc["A", "rank_in_group"] == 1 and s.loc["D", "rank_in_group"] == 4


def test_small_group_is_scored_but_flagged_thin():
    three = metrics_for(fund("A", market_cap=1000.0), fund("B", market_cap=2000.0), fund("C", market_cap=3000.0))
    s = score(rank_metrics(three, CFG), three, CFG)
    assert s["thin_group"].all() and s["composite"].notna().all()


# ---- text views --------------------------------------------------------

def test_text_views_run_and_name_the_fallback():
    frame = group()
    frame.loc[frame["ticker"] == "B", "ebitda_yield"] = np.nan
    ranks = rank_metrics(frame, CFG)
    s = score(ranks, frame, CFG)
    table = group_table(s, frame, "g")
    assert table.splitlines()[0] == "g - 4 companies" and len(table.splitlines()) == 6
    detail = company_detail("B", s, ranks)
    assert "(used ebit_yield)" in detail and "rank" in detail
    assert "no peer group" in company_detail("X", s, ranks)
    assert "not in the scores" in company_detail("ZZZ", s, ranks)
