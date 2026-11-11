import pandas as pd

from screener.cleaning.facts import RAW_COLUMNS, dedupe, flag_scale_errors

SHARES = "WeightedAverageNumberOfDilutedSharesOutstanding"


def row(val, year, filed, accn, tag=SHARES, unit="shares", form="10-K", cik="0000000001"):
    return {
        "cik10": cik, "taxonomy": "us-gaap", "tag": tag, "unit": unit,
        "start": pd.Timestamp(f"{year}-01-01"), "end": pd.Timestamp(f"{year}-12-31"),
        "val": float(val), "accn": accn, "form": form, "filed": pd.Timestamp(filed),
        "fy": year, "fp": "FY", "frame": None,
    }


def frame(rows):
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def history():
    """Five years of a ~37M share count, each reported once, correctly."""
    return [row(36e6 + i * 1e6, 2015 + i, f"{2016 + i}-03-01", f"K-{2015 + i}") for i in range(5)]


def fact(facts, year, tag=SHARES):
    rows = facts[(facts["tag"] == tag) & (facts["end"] == pd.Timestamp(f"{year}-12-31"))]
    assert len(rows) == 1
    return rows.iloc[0]


def test_later_filing_in_the_wrong_scale_does_not_win():
    # MPWR-style: the next year's 10-K repeats 2019 in thousands.
    raw = frame(history() + [row(40_000, 2019, "2021-03-01", "K-2020")])
    assert flag_scale_errors(raw).tolist() == [False] * 5 + [True]
    got = fact(dedupe(raw), 2019)
    assert (got["val_first"], got["val_latest"], got["accn_latest"]) == (40e6, 40e6, "K-2019")
    assert (got["restated"], got["scale_errors"], got["large_disagreement"]) == (False, 1, False)


def test_first_filing_in_the_wrong_scale_is_the_one_set_aside():
    rows = history()
    rows[4] = row(40_000, 2019, "2020-03-01", "K-2019")  # original filing was wrong
    raw = frame(rows + [row(40e6, 2019, "2021-03-01", "K-2020")])
    got = fact(dedupe(raw), 2019)
    assert (got["val_first"], got["val_latest"], got["accn_first"]) == (40e6, 40e6, "K-2020")
    assert got["scale_errors"] == 1 and got["n_filings"] == 1


def test_rounded_wrong_scale_values_still_match():
    # 35,673,280 shares later reported as 35,673 (thousands, rounded).
    raw = frame(history() + [row(35_673_280, 2020, "2021-03-01", "A"), row(35_673, 2020, "2022-03-01", "B")])
    got = fact(dedupe(raw), 2020)
    assert (got["val_latest"], got["scale_errors"]) == (35_673_280.0, 1)


def test_wrong_scale_by_a_million_and_for_small_decimals():
    eps = [row(1.0 + i / 10, 2015 + i, f"{2016 + i}-03-01", f"K-{2015 + i}", tag="EarningsPerShareDiluted", unit="USD/shares") for i in range(5)]
    eps.append(row(1400.0, 2019, "2021-03-01", "K-2020", tag="EarningsPerShareDiluted", unit="USD/shares"))
    raw = frame(eps + history() + [row(40.0, 2019, "2021-03-01", "K-2020")])
    facts = dedupe(raw)
    assert fact(facts, 2019, "EarningsPerShareDiluted")["val_latest"] == 1.4
    assert fact(facts, 2019)["val_latest"] == 40e6


def test_every_filing_repeating_the_wrong_value_is_set_aside():
    raw = frame(history() + [row(40_000, 2019, "2021-03-01", "K-2020"), row(40_000, 2019, "2022-03-01", "K-2021")])
    got = fact(dedupe(raw), 2019)
    assert (got["val_latest"], got["scale_errors"]) == (40e6, 2)


def test_a_real_restatement_is_left_alone():
    raw = frame(history() + [row(31e6, 2019, "2021-03-01", "K-2020")])
    assert not flag_scale_errors(raw).any()
    got = fact(dedupe(raw), 2019)
    assert (got["val_first"], got["val_latest"], got["restated"], got["scale_errors"]) == (40e6, 31e6, True, 0)


def test_a_lone_outlier_is_not_touched():
    # Only one filing reports 2020, in the wrong scale: nothing corroborates a fix.
    raw = frame(history() + [row(41_000, 2020, "2021-03-01", "K-2020")])
    assert not flag_scale_errors(raw).any()
    assert fact(dedupe(raw), 2020)["val_latest"] == 41_000.0


def test_opposite_signs_are_never_a_scale_match():
    rows = [row(50e6 + i, 2015 + i, f"{2016 + i}-03-01", f"K-{2015 + i}", tag="NetIncomeLoss", unit="USD") for i in range(5)]
    rows.append(row(-50_000.004, 2019, "2021-03-01", "K-2020", tag="NetIncomeLoss", unit="USD"))
    raw = frame(rows)
    assert not flag_scale_errors(raw).any()
    got = fact(dedupe(raw), 2019, "NetIncomeLoss")
    assert got["restated"] and got["large_disagreement"]


def test_large_non_scale_disagreement_is_flagged_but_not_resolved():
    # MCHP-style: a later 10-K tags an unrelated number as equity (184x apart).
    rows = [row(2.0e9 + i * 1e8, 2015 + i, f"{2016 + i}-03-01", f"K-{2015 + i}", tag="StockholdersEquity", unit="USD") for i in range(5)]
    rows.append(row(11.076e6, 2019, "2021-03-01", "K-2020", tag="StockholdersEquity", unit="USD"))
    got = fact(dedupe(frame(rows)), 2019, "StockholdersEquity")
    assert (got["val_first"], got["val_latest"]) == (2.4e9, 11.076e6)
    assert (got["large_disagreement"], got["scale_errors"]) == (True, 0)


def test_typical_size_is_per_company_and_per_tag():
    # Company 2 really does have ~40 thousand shares; company 1's history must not flag it.
    small = [row(38_000 + i * 1000, 2015 + i, f"{2016 + i}-03-01", f"S-{2015 + i}", cik="0000000002") for i in range(5)]
    small.append(row(42e6, 2019, "2021-03-01", "S-2020", cik="0000000002"))  # this one is the error
    raw = frame(history() + small)
    facts = dedupe(raw)
    got = facts[(facts["cik10"] == "0000000002") & (facts["end"] == pd.Timestamp("2019-12-31"))].iloc[0]
    assert (got["val_latest"], got["scale_errors"]) == (42_000.0, 1)


def test_proxy_values_are_ignored_when_judging_scale():
    raw = frame(history() + [row(40_000, 2019, "2021-04-01", "PX", form="DEF 14A")])
    assert not flag_scale_errors(raw).any()
    got = fact(dedupe(raw), 2019)
    assert (got["val_latest"], got["n_filings"], got["scale_errors"]) == (40e6, 1, 0)


def test_zero_values_do_not_break_anything():
    raw = frame(history() + [row(0, 2019, "2021-03-01", "K-2020")])
    assert not flag_scale_errors(raw).any()
    got = fact(dedupe(raw), 2019)
    assert (got["val_latest"], got["restated"], got["large_disagreement"]) == (0.0, True, False)
