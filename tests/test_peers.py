import numpy as np
import pandas as pd
import pytest

from screener import config
from screener.data.tickers import read_universe
from screener.scoring.peers import (
    PeerGroupError,
    check_against_universe,
    group_affinity,
    load_peer_groups,
    residual_returns,
    weekly_returns,
)


def write(tmp_path, text):
    path = tmp_path / "peers.csv"
    path.write_text(text)
    return path


def test_real_peer_file_covers_the_universe_exactly():
    groups = load_peer_groups()
    missing, extra = check_against_universe(groups, read_universe(config.DEFAULT_UNIVERSE))
    assert (missing, extra) == ([], [])
    sizes = groups.dropna(subset=["peer_group"]).groupby("peer_group").size()
    assert (sizes >= 2).all()  # a group of one cannot be ranked or paired


def test_load_normalizes_tickers_and_blank_group_is_unplaced(tmp_path):
    groups = load_peer_groups(write(tmp_path, "# note\nticker,peer_group\nnvda,compute\nbrk.b, \nAMD,compute\n"))
    assert groups["ticker"].tolist() == ["NVDA", "BRK-B", "AMD"]
    assert groups["peer_group"].tolist()[0] == "compute"
    assert groups["peer_group"].isna().tolist() == [False, True, False]


@pytest.mark.parametrize("text, message", [
    ("ticker,group\nNVDA,a\n", "columns"),
    ("ticker,peer_group\nNVDA,a\nnvda,b\n", "more than once"),
])
def test_load_rejects_bad_files(tmp_path, text, message):
    with pytest.raises(PeerGroupError, match=message):
        load_peer_groups(write(tmp_path, text))


def test_check_against_universe():
    groups = pd.DataFrame({"ticker": ["A", "B", "X"], "peer_group": ["g", "g", None]})
    assert check_against_universe(groups, ["A", "B", "C"]) == (["C"], ["X"])


def prices(closes, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"date": dates, "adj_close": closes})


def test_weekly_returns_use_the_last_close_of_each_week():
    # Two full weeks of daily closes: week 1 ends at 104, week 2 at 109.2.
    frame = prices([100, 101, 102, 103, 104, 105, 106, 107, 108, 109.2])
    returns = weekly_returns({"A": frame})
    assert returns.index.tolist() == [pd.Timestamp("2024-01-12")]
    assert returns["A"].iloc[0] == pytest.approx(109.2 / 104 - 1)


def test_weekly_returns_respect_as_of_and_window():
    frame = prices(list(np.linspace(100, 200, 100)))
    cut = weekly_returns({"A": frame}, as_of=pd.Timestamp("2024-03-01"))
    assert cut.index.max() <= pd.Timestamp("2024-03-01")
    assert len(weekly_returns({"A": frame}, weeks=5)) == 5


def synthetic_returns(weeks=200, seed=0):
    """Six companies driven by one sector factor plus one of two group factors."""
    rng = np.random.default_rng(seed)
    sector = rng.normal(0, 0.03, weeks)
    factor = {"g1": rng.normal(0, 0.02, weeks), "g2": rng.normal(0, 0.02, weeks)}
    membership = {"A": "g1", "B": "g1", "C": "g1", "D": "g2", "E": "g2", "F": "g2"}
    index = pd.date_range("2022-01-07", periods=weeks, freq="W-FRI")
    data = {t: sector + factor[g] + rng.normal(0, 0.01, weeks) for t, g in membership.items()}
    return pd.DataFrame(data, index=index), membership


def test_residuals_remove_the_sector_move():
    returns, _ = synthetic_returns()
    raw = returns.corr()
    residual = residual_returns(returns).corr()
    # Before: everything is positively correlated because of the sector.
    assert raw.loc["A", "D"] > 0.3
    # After: same-group pairs stay correlated, cross-group pairs turn negative or flat.
    assert residual.loc["A", "B"] > 0.3
    assert residual.loc["A", "D"] < 0


def test_company_with_too_little_history_is_left_out():
    returns, _ = synthetic_returns()
    returns.loc[returns.index[:-30], "F"] = np.nan  # only 30 weeks of data
    assert "F" not in residual_returns(returns).columns


def test_affinity_agrees_with_correct_groups_and_flags_a_wrong_one():
    returns, membership = synthetic_returns()
    correlation = residual_returns(returns).corr()

    right = pd.DataFrame({"ticker": list(membership), "peer_group": list(membership.values())})
    assert group_affinity(correlation, right)["agrees"].tolist() == [True] * 6

    wrong = right.copy()
    wrong.loc[wrong["ticker"] == "C", "peer_group"] = "g2"  # C really moves with g1
    result = group_affinity(correlation, wrong).set_index("ticker")
    assert result.loc["C", "agrees"] is False or result.loc["C", "agrees"] == False  # noqa: E712
    assert result.loc["C", "best_group"] == "g1"
    assert result.loc["C", "closest"].split(",")[0].split()[0] in {"A", "B"}


def test_affinity_for_an_unplaced_company_suggests_a_group():
    returns, membership = synthetic_returns()
    correlation = residual_returns(returns).corr()
    groups = pd.DataFrame({"ticker": list(membership), "peer_group": ["g1", "g1", None, "g2", "g2", "g2"]})
    row = group_affinity(correlation, groups).set_index("ticker").loc["C"]
    assert row["peer_group"] is None and row["agrees"] is None
    assert row["best_group"] == "g1" and pd.isna(row["own_corr"])
