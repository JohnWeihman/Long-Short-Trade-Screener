import numpy as np
import pandas as pd
import pytest

from screener.pairing.build import evaluate_pairs
from screener.pairing.candidates import COLUMNS, candidate_pairs
from screener.pairing.config import PairingConfig, PairingConfigError, load_pairing_config, parse_pairing_config
from screener.pairing.report import pair_detail, pairs_table
from screener.pairing.spread import daily_returns, spread_series, spread_stats
from screener.pairing.stats import beta, betas, correlation, equal_weight_benchmark, hedge_ratio, share_ratio

CFG = PairingConfig(min_overlap_weeks=20, correlation_window_weeks=60, spread_history_days=100, spread_zscore_window_days=40)


def score_row(ticker, composite, valuation, quality, direction, group="g", thin=False):
    return {"ticker": ticker, "peer_group": group, "composite": composite, "valuation": valuation,
            "quality": quality, "direction": direction, "thin_group": thin}


def scores():
    return pd.DataFrame([
        score_row("AAA", 0.80, 0.9, 0.8, 0.7),
        score_row("BBB", 0.50, 0.2, 0.9, 0.4),
        score_row("CCC", 0.20, 0.1, 0.2, 0.3),
        score_row("DDD", 0.60, 0.6, 0.6, 0.6, group="h", thin=True),
        score_row("EEE", 0.30, 0.3, 0.3, 0.3, group="h", thin=True),
        score_row("ZZZ", None, None, None, None),            # in the group but not scored
        score_row("MU", 0.9, 0.9, 0.9, 0.9, group=None),      # unplaced
    ])


# ---- config ------------------------------------------------------------

def test_real_pairing_config_loads():
    cfg = load_pairing_config()
    assert (cfg.min_composite_gap, cfg.min_correlation, cfg.min_pillars_favoring_long) == (0.25, 0.50, 2)
    assert cfg.beta_benchmark == "universe_equal_weight" and cfg.correlation_window_weeks == 104


@pytest.mark.parametrize("raw, message", [
    ({"min_gap": 0.3}, "unknown"),
    ({"beta_benchmark": "SPY"}, "beta_benchmark"),
    ({"min_composite_gap": 1.5}, "min_composite_gap"),
    ({"min_overlap_weeks": 200, "correlation_window_weeks": 104}, "min_overlap_weeks"),
])
def test_bad_pairing_config_is_rejected(raw, message):
    with pytest.raises(PairingConfigError, match=message):
        parse_pairing_config(raw)


# ---- candidates --------------------------------------------------------

def test_candidates_are_within_group_long_the_higher_score():
    pairs = candidate_pairs(scores(), CFG)
    assert list(pairs.columns) == COLUMNS
    assert pairs["pair_id"].tolist() == ["AAA/CCC", "AAA/BBB", "BBB/CCC", "DDD/EEE"]  # widest gap first
    assert not pairs["pair_id"].str.contains("ZZZ|MU").any()
    first = pairs.iloc[0]
    assert (first["long"], first["short"], first["peer_group"]) == ("AAA", "CCC", "g")
    assert first["gap"] == pytest.approx(0.60)
    assert (first["gap_valuation"], first["gap_quality"], first["gap_direction"]) == pytest.approx((0.8, 0.6, 0.4))


def test_gap_and_pillar_rules_are_recorded_not_applied():
    pairs = candidate_pairs(scores(), CFG).set_index("pair_id")
    assert len(pairs) == 4  # nothing dropped
    assert pairs["passes_gap"].to_dict() == {"AAA/CCC": True, "AAA/BBB": True, "BBB/CCC": True, "DDD/EEE": True}
    # AAA beats BBB on valuation and direction but not quality: two pillars, enough.
    assert pairs.loc["AAA/BBB", "pillars_favoring_long"] == 2 and pairs.loc["AAA/BBB", "passes_pillars"]
    strict = candidate_pairs(scores(), PairingConfig(min_composite_gap=0.35, min_pillars_favoring_long=3)).set_index("pair_id")
    assert strict["passes_gap"].to_dict() == {"AAA/CCC": True, "AAA/BBB": False, "BBB/CCC": False, "DDD/EEE": False}
    assert not strict.loc["AAA/BBB", "passes_pillars"] and strict.loc["AAA/CCC", "passes_pillars"]


def test_gap_exactly_at_the_threshold_passes_and_thin_group_is_carried():
    pairs = candidate_pairs(scores(), PairingConfig(min_composite_gap=0.30)).set_index("pair_id")
    assert pairs.loc["AAA/BBB", "passes_gap"] and pairs.loc["DDD/EEE", "passes_gap"]  # 0.80 - 0.50 in floats
    assert pairs.loc["DDD/EEE", "thin_group"] and not pairs.loc["AAA/BBB", "thin_group"]


def test_missing_pillar_does_not_count_for_the_long():
    table = scores()
    table.loc[table["ticker"] == "CCC", "direction"] = None
    pairs = candidate_pairs(table, CFG).set_index("pair_id")
    assert pairs.loc["AAA/CCC", "pillars_favoring_long"] == 2 and pd.isna(pairs.loc["AAA/CCC", "gap_direction"])


# ---- statistics --------------------------------------------------------

def weekly(n=60, seed=1):
    """A, B, C share a market factor with betas 1.5, 1.0, 0.5; N is pure noise; S is short-lived."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.03, n)
    index = pd.date_range("2025-01-03", periods=n, freq="W-FRI")
    frame = pd.DataFrame({
        "AAA": 1.5 * market + rng.normal(0, 0.004, n),
        "BBB": 1.0 * market + rng.normal(0, 0.004, n),
        "CCC": 0.5 * market + rng.normal(0, 0.004, n),
        "NNN": rng.normal(0, 0.03, n),
    }, index=index)
    frame["SSS"] = frame["BBB"].where(frame.index >= index[-10])  # only 10 weeks of history
    return frame, pd.Series(market, index=index)


def test_correlation_and_overlap():
    frame, _ = weekly()
    corr, n = correlation(frame["AAA"], frame["BBB"], 20)
    assert corr > 0.95 and n == 60
    low, _ = correlation(frame["AAA"], frame["NNN"], 20)
    assert abs(low) < 0.4
    short, n = correlation(frame["AAA"], frame["SSS"], 20)
    assert np.isnan(short) and n == 10
    flat, _ = correlation(frame["AAA"], pd.Series(0.0, index=frame.index), 20)
    assert np.isnan(flat)


def test_beta_recovers_the_true_slope():
    frame, market = weekly()
    got = betas(frame[["AAA", "BBB", "CCC"]], market, 20)
    assert got["AAA"] == pytest.approx(1.5, abs=0.1)
    assert got["BBB"] == pytest.approx(1.0, abs=0.1)
    assert got["CCC"] == pytest.approx(0.5, abs=0.1)
    assert np.isnan(beta(frame["SSS"], market, 20))
    assert np.isnan(beta(frame["AAA"], pd.Series(0.01, index=frame.index), 20))  # benchmark never moves


def test_beta_matches_the_textbook_formula():
    a = pd.Series([0.02, -0.01, 0.03, 0.00, -0.02])
    m = pd.Series([0.01, -0.01, 0.02, 0.01, -0.01])
    assert beta(a, m, 3) == pytest.approx(np.cov(a, m, ddof=1)[0, 1] / np.var(m, ddof=1))


def test_equal_weight_benchmark_skips_missing_values():
    frame = pd.DataFrame({"A": [0.02, 0.04], "B": [0.04, np.nan]})
    assert equal_weight_benchmark(frame).tolist() == pytest.approx([0.03, 0.04])


def test_hedge_ratio_and_share_ratio():
    assert hedge_ratio(1.5, 1.0) == 1.5
    assert hedge_ratio(0.6, 1.2) == 0.5
    assert np.isnan(hedge_ratio(1.0, 0.0)) and np.isnan(hedge_ratio(1.0, -0.3)) and np.isnan(hedge_ratio(np.nan, 1.0))
    # $1.50 short per $1 long; long at $200, short at $50 -> 6 short shares per long share.
    assert share_ratio(1.5, 200.0, 50.0) == pytest.approx(6.0)
    assert np.isnan(share_ratio(np.nan, 200.0, 50.0)) and np.isnan(share_ratio(1.5, 200.0, np.nan))


# ---- spread ------------------------------------------------------------

def test_daily_returns_respect_as_of():
    prices = pd.DataFrame({"date": pd.bdate_range("2026-01-05", periods=4), "adj_close": [100.0, 110.0, 99.0, 99.0]})
    got = daily_returns(prices)
    assert got.tolist() == pytest.approx([0.10, -0.10, 0.0])
    assert len(daily_returns(prices, as_of=pd.Timestamp("2026-01-06"))) == 1


def test_spread_is_long_minus_hedged_short():
    dates = pd.bdate_range("2026-01-05", periods=3)
    long_r = pd.Series([0.02, 0.01, -0.01], index=dates)
    short_r = pd.Series([0.01, 0.02, 0.00], index=dates)
    spread = spread_series("L/S", long_r, short_r, 1.5, 100)
    assert spread["spread_return"].tolist() == pytest.approx([0.005, -0.02, -0.01])
    assert spread["spread"].tolist() == pytest.approx([0.005, -0.015, -0.025])
    assert spread["pair_id"].unique().tolist() == ["L/S"]


def test_spread_uses_only_days_both_traded_and_the_history_limit():
    dates = pd.bdate_range("2026-01-05", periods=5)
    long_r = pd.Series([0.01] * 5, index=dates)
    short_r = pd.Series([0.01, np.nan, 0.01, 0.01], index=dates[:4])
    assert len(spread_series("L/S", long_r, short_r, 1.0, 100)) == 3
    assert len(spread_series("L/S", long_r, short_r, 1.0, 2)) == 2
    assert spread_series("L/S", long_r, short_r, np.nan, 100).empty


def test_spread_stats():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2025-01-01", periods=300)
    flat = pd.Series(rng.normal(0, 0.001, 300), index=dates)
    long_r = flat.copy()
    long_r.iloc[-5:] += 0.02  # the long jumps in the last week
    spread = spread_series("L/S", long_r, pd.Series(0.0, index=dates), 1.0, 300)
    stats = spread_stats(spread, 252)
    assert stats["spread_z"] > 2
    assert stats["spread_chg_1m"] == pytest.approx(spread["spread_return"].tail(21).sum())
    assert not np.isnan(stats["spread_chg_12m"])
    short = spread_stats(spread.tail(30), 252)
    assert np.isnan(short["spread_z"]) and np.isnan(short["spread_chg_3m"]) and not np.isnan(short["spread_chg_1m"])
    assert np.isnan(spread_stats(spread.iloc[0:0], 252)["spread_z"])


# ---- evaluation end to end ---------------------------------------------

def evaluated(cfg=CFG, table=None):
    frame, _ = weekly()
    table = scores() if table is None else table
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2025-06-02", periods=120)
    daily = {t: pd.Series(rng.normal(0, 0.01, 120), index=dates) for t in ("AAA", "BBB", "CCC", "NNN")}
    prices = {"AAA": 200.0, "BBB": 50.0, "CCC": 25.0, "NNN": 10.0}
    notes = {"AAA": "", "CCC": "no D&A: EBITDA unavailable"}
    return evaluate_pairs(candidate_pairs(table, cfg), frame, daily, prices, notes, cfg)


def test_passing_pairs_are_ranked_and_get_hedge_ratios():
    pairs, spreads = evaluated()
    by_id = pairs.set_index("pair_id")
    assert by_id.loc["AAA/CCC", "passes"] and by_id.loc["AAA/CCC", "rank"] == 1
    assert by_id.loc["AAA/BBB", "rank"] == 2 and by_id.loc["BBB/CCC", "rank"] == 3
    # Betas against the three-plus-noise universe index differ from 1.5/1.0/0.5, but their ratios hold.
    assert by_id.loc["AAA/CCC", "hedge_ratio"] == pytest.approx(3.0, rel=0.1)
    assert by_id.loc["AAA/BBB", "hedge_ratio"] == pytest.approx(1.5, rel=0.1)
    assert by_id.loc["AAA/BBB", "share_ratio"] == pytest.approx(by_id.loc["AAA/BBB", "hedge_ratio"] * 200 / 50)
    assert by_id.loc["AAA/BBB", "pair_beta"] == pytest.approx(1.5, rel=0.1)
    assert by_id.loc["AAA/CCC", "notes"] == "CCC: no D&A: EBITDA unavailable"
    assert set(spreads["pair_id"]) == {"AAA/CCC", "AAA/BBB", "BBB/CCC"}
    assert len(spreads[spreads["pair_id"] == "AAA/BBB"]) == 100  # spread_history_days


def test_pairs_without_price_history_fail_with_a_reason_but_stay_listed():
    pairs, spreads = evaluated()
    row = pairs.set_index("pair_id").loc["DDD/EEE"]  # no returns for either
    assert not row["passes"] and pd.isna(row["rank"])
    assert "too little shared price history" in row["reasons"] and "no usable hedge ratio" in row["reasons"]
    assert row["notes"] == "thin peer group"
    assert "DDD/EEE" not in set(spreads["pair_id"])
    assert len(pairs) == 4


def test_low_correlation_pair_fails():
    table = pd.DataFrame([score_row("AAA", 0.8, 0.9, 0.8, 0.7), score_row("NNN", 0.2, 0.1, 0.2, 0.3)])
    pairs, _ = evaluated(table=table)
    row = pairs.iloc[0]
    assert not row["passes"] and not row["passes_correlation"] and "correlation" in row["reasons"]


def test_score_rules_show_up_in_the_reasons():
    strict = PairingConfig(min_composite_gap=0.35, min_pillars_favoring_long=3, min_overlap_weeks=20,
                           correlation_window_weeks=60, spread_history_days=100, spread_zscore_window_days=40)
    pairs, _ = evaluated(cfg=strict)
    row = pairs.set_index("pair_id").loc["AAA/BBB"]
    assert "score gap 0.30 below 0.35" in row["reasons"] and "only 2 pillar(s) favor the long" in row["reasons"]
    assert pairs["passes"].sum() == 1  # only AAA/CCC


# ---- text views --------------------------------------------------------

def test_text_views():
    pairs, _ = evaluated()
    pairs["as_of"] = pd.Timestamp("2026-02-15")
    passing = pairs_table(pairs)
    assert "AAA   CCC" in passing and "DDD" not in passing
    everything = pairs_table(pairs, include_failing=True)
    assert "FAILS: too little shared price history" in everything
    assert pairs_table(pairs[~pairs["passes"]]) == "No pairs pass the filters."

    metric_columns = ["ev_ebitda", "ev_ebit", "pe", "fcf_yield", "ev_sales", "gross_margin", "operating_margin",
                      "roic", "net_debt_to_ebitda", "revenue_growth", "operating_margin_change"]
    metrics = pd.DataFrame([{"ticker": t, **{c: 0.5 for c in metric_columns}} for t in ("AAA", "BBB", "CCC")])
    detail = pair_detail(pairs.set_index("pair_id").loc["AAA/BBB"].rename(None), scores(), metrics)
    assert detail.startswith("LONG AAA / SHORT BBB - g - rank 2")
    assert "Hedge ratio" in detail and "BBB shares per AAA share" in detail
