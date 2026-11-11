import pandas as pd
import pytest

from screener.data.prices import PRICE_COLUMNS, PriceDataError, fetch_prices, load_prices, normalize_yfinance


def yf_frame():
    """Shaped like yfinance Ticker.history(auto_adjust=False): tz-aware index,
    title-case columns, deliberately out of order with one empty row."""
    index = pd.DatetimeIndex(
        ["2024-06-11 00:00:00", "2024-06-10 00:00:00", "2024-06-12 00:00:00"],
        tz="America/New_York",
        name="Date",
    )
    return pd.DataFrame(
        {
            "Open": [121.0, 120.0, None],
            "High": [123.0, 122.0, None],
            "Low": [119.0, 118.0, None],
            "Close": [122.0, 121.0, None],
            "Adj Close": [121.5, 120.5, None],
            "Volume": [2000, 1000, 0],
            "Dividends": [0.0, 0.01, 0.0],
            "Stock Splits": [0.0, 10.0, 0.0],
        },
        index=index,
    )


def test_normalize_gives_the_standard_columns_sorted_by_date():
    out = normalize_yfinance(yf_frame(), "nvda")
    assert list(out.columns) == PRICE_COLUMNS
    assert out["date"].tolist() == [pd.Timestamp("2024-06-10"), pd.Timestamp("2024-06-11")]
    assert out["ticker"].unique().tolist() == ["NVDA"]


def test_normalize_keeps_values_on_their_own_dates():
    out = normalize_yfinance(yf_frame(), "NVDA").set_index("date")
    first = out.loc[pd.Timestamp("2024-06-10")]
    assert (first["close"], first["adj_close"], first["volume"]) == (121.0, 120.5, 1000)
    assert (first["dividends"], first["splits"]) == (0.01, 10.0)


def test_normalize_keeps_the_local_calendar_date():
    # Midnight New York is 04:00 UTC; converting to UTC first would be harmless
    # here, but an evening timestamp must not roll to the next day.
    index = pd.DatetimeIndex(["2024-06-10 20:00:00"], tz="America/New_York")
    raw = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1]}, index=index)
    out = normalize_yfinance(raw, "X")
    assert out["date"].tolist() == [pd.Timestamp("2024-06-10")]


def test_normalize_fills_columns_yfinance_may_omit():
    raw = yf_frame().drop(columns=["Adj Close", "Dividends", "Stock Splits"])
    out = normalize_yfinance(raw, "NVDA")
    assert out["adj_close"].isna().all()
    assert (out["dividends"] == 0).all() and (out["splits"] == 0).all()


def test_normalize_rejects_empty_or_malformed_data():
    with pytest.raises(PriceDataError):
        normalize_yfinance(pd.DataFrame(), "NVDA")
    with pytest.raises(PriceDataError, match="missing columns"):
        normalize_yfinance(yf_frame().drop(columns=["Close"]), "NVDA")


class FakeSource:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def history(self, ticker):
        self.calls += 1
        return normalize_yfinance(yf_frame(), ticker)


def test_prices_are_cached_and_only_refetched_on_refresh(cache):
    source = FakeSource()
    assert fetch_prices(source, cache, "NVDA").status == "downloaded"
    assert fetch_prices(source, cache, "NVDA").status == "cached"
    assert source.calls == 1
    assert fetch_prices(source, cache, "NVDA", refresh=True).status == "downloaded"
    assert source.calls == 2

    saved = load_prices(cache, "NVDA", source_name="fake")
    assert list(saved.columns) == PRICE_COLUMNS
    assert len(saved) == 2
