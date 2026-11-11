import pandas as pd
import pytest

from screener.cleaning.facts import RAW_COLUMNS, dedupe
from screener.cleaning.splits import adjust_for_splits, load_splits, split_factor

A, B = "0000000001", "0000000002"


def splits(*rows):
    return pd.DataFrame(
        [{"cik10": c, "split_date": pd.Timestamp(d), "ratio": float(r)} for c, d, r in rows],
        columns=["cik10", "split_date", "ratio"],
    )


# Company A: 4-for-1 in July 2021, 10-for-1 in June 2024. Company B: 1-for-5 reverse split.
SPLITS = splits((A, "2021-07-20", 4), (A, "2024-06-10", 10), (B, "2023-03-01", 0.2))


def factors(cik, dates, table=SPLITS):
    n = len(dates)
    return split_factor(pd.Series([cik] * n), pd.Series(pd.to_datetime(dates)), table).tolist()


def test_factor_is_the_product_of_all_later_splits():
    assert factors(A, ["2021-05-01", "2022-05-01", "2025-01-01"]) == [40.0, 10.0, 1.0]


def test_a_value_stated_on_the_split_date_is_already_post_split():
    assert factors(A, ["2024-06-09", "2024-06-10", "2024-06-11"]) == [10.0, 1.0, 1.0]


def test_reverse_split_and_company_without_splits():
    assert factors(B, ["2023-01-01", "2023-06-01"]) == [0.2, 1.0]
    assert factors("0000000003", ["2020-01-01"]) == [1.0]
    assert factors(A, ["2020-01-01"], table=splits()) == [1.0]


def test_factor_keeps_rows_in_place_when_companies_are_interleaved():
    cik = pd.Series([A, B, A, B], index=[10, 11, 12, 13])
    dates = pd.Series(pd.to_datetime(["2022-01-01", "2022-01-01", "2025-01-01", "2024-01-01"]), index=cik.index)
    assert split_factor(cik, dates, SPLITS).to_dict() == {10: 10.0, 11: 0.2, 12: 1.0, 13: 1.0}


def raw_row(tag, unit, val, filed, *, taxonomy="us-gaap", start="2023-01-30", end="2024-01-28", accn="X", cik=A):
    return {
        "cik10": cik, "taxonomy": taxonomy, "tag": tag, "unit": unit,
        "start": pd.Timestamp(start) if start else pd.NaT, "end": pd.Timestamp(end),
        "val": float(val), "accn": accn, "form": "10-K", "filed": pd.Timestamp(filed),
        "fy": 2024, "fp": "FY", "frame": None,
    }


def frame(rows):
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def test_shares_are_multiplied_per_share_divided_and_money_left_alone():
    raw = frame([
        raw_row("DilutedShares", "shares", 2_494e6, "2024-02-21"),
        raw_row("EpsDiluted", "USD/shares", 11.93, "2024-02-21"),
        raw_row("Revenues", "USD", 60_922e6, "2024-02-21"),
    ])
    out = adjust_for_splits(raw, SPLITS)
    assert out["val"].tolist() == pytest.approx([24_940e6, 1.193, 60_922e6])
    assert out["val_reported"].tolist() == [2_494e6, 11.93, 60_922e6]
    assert out["split_factor"].tolist() == [10.0, 10.0, 1.0]


def test_filing_after_the_split_is_left_alone():
    raw = frame([raw_row("DilutedShares", "shares", 24_940e6, "2025-02-26")])
    out = adjust_for_splits(raw, SPLITS)
    assert (out.loc[0, "val"], out.loc[0, "split_factor"]) == (24_940e6, 1.0)


def test_cover_page_count_goes_by_its_own_date_not_the_filing_date():
    # Counted on 7 June (pre-split) but the filing went in after the 10 June split.
    cover = raw_row("EntityCommonStockSharesOutstanding", "shares", 2_460e6, "2024-06-12",
                    taxonomy="dei", start=None, end="2024-06-07")
    # A balance-sheet count in the same filing has been restated by the company already.
    balance = raw_row("CommonStockSharesOutstanding", "shares", 24_600e6, "2024-06-12", start=None, end="2024-04-28")
    out = adjust_for_splits(frame([cover, balance]), SPLITS)
    assert out["val"].tolist() == [24_600e6, 24_600e6]
    assert out["split_factor"].tolist() == [10.0, 1.0]


def test_pre_and_post_split_filings_of_one_period_agree_after_adjustment():
    raw = frame([
        raw_row("DilutedShares", "shares", 2_494e6, "2024-02-21", accn="K-24"),
        raw_row("DilutedShares", "shares", 24_940e6, "2025-02-26", accn="K-25"),
        raw_row("EpsDiluted", "USD/shares", 11.93, "2024-02-21", accn="K-24"),
        raw_row("EpsDiluted", "USD/shares", 1.19, "2025-02-26", accn="K-25"),
    ])
    before = dedupe(raw).set_index("tag")
    assert before.loc["DilutedShares", "restated"] and before.loc["DilutedShares", "large_disagreement"]

    after = dedupe(adjust_for_splits(raw, SPLITS)).set_index("tag")
    shares = after.loc["DilutedShares"]
    assert (shares["val_first"], shares["val_latest"]) == (24_940e6, 24_940e6)
    assert not shares["restated"] and not shares["large_disagreement"]
    # EPS was rounded to cents after the split: 1.193 vs 1.19 is a rounding difference, not 10x.
    eps = after.loc["EpsDiluted"]
    assert eps["val_first"] == pytest.approx(1.193) and eps["val_latest"] == 1.19
    assert not eps["large_disagreement"]


def test_no_splits_changes_nothing():
    raw = frame([raw_row("DilutedShares", "shares", 100.0, "2024-02-21")])
    out = adjust_for_splits(raw, splits())
    assert (out.loc[0, "val"], out.loc[0, "split_factor"], out.loc[0, "val_reported"]) == (100.0, 1.0, 100.0)


def test_load_splits_reads_the_price_cache(cache):
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2024-06-07", "2024-06-10", "2024-06-11"]),
        "splits": [0.0, 10.0, 0.0],
    })
    path = cache.prices_path("yfinance", "NVDA")
    path.parent.mkdir(parents=True)
    prices.to_parquet(path, index=False)

    table = load_splits(cache, {A: "NVDA", B: "NOFILE"})
    assert table.to_dict("records") == [{"cik10": A, "split_date": pd.Timestamp("2024-06-10"), "ratio": 10.0}]
    assert load_splits(cache, {}).empty
