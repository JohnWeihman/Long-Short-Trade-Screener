import pytest

from screener.data.tickers import (
    TICKER_MAP_URL,
    UnknownTickerError,
    fetch_ticker_map,
    load_ticker_map,
    parse_ticker_map,
    read_universe,
    resolve_tickers,
)

from conftest import FakeFetcher

RAW = {
    "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "2": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
    "3": {"cik_str": 2488, "ticker": "amd", "title": "ADVANCED MICRO DEVICES INC"},
    "4": {"cik_str": 999999, "ticker": "AAPL", "title": "Duplicate ticker, smaller company"},
}


def test_parse_pads_ciks_and_normalizes_tickers():
    frame = parse_ticker_map(RAW)
    assert list(frame.columns) == ["ticker", "cik", "cik10", "name"]
    by_ticker = frame.set_index("ticker")
    assert by_ticker.loc["AMD", "cik10"] == "0000002488"
    assert by_ticker.loc["AMD", "cik"] == 2488
    assert by_ticker.loc["NVDA", "cik10"] == "0001045810"


def test_parse_keeps_first_row_for_a_repeated_ticker():
    frame = parse_ticker_map(RAW)
    assert (frame["ticker"] == "AAPL").sum() == 1
    assert frame.set_index("ticker").loc["AAPL", "cik"] == 320193


def test_resolve_keeps_request_order_and_accepts_dot_class_tickers():
    frame = parse_ticker_map(RAW)
    resolved = resolve_tickers(frame, ["brk.b", " nvda "])
    assert resolved["ticker"].tolist() == ["BRK-B", "NVDA"]
    assert resolved["cik10"].tolist() == ["0001067983", "0001045810"]


def test_resolve_names_every_unknown_ticker():
    with pytest.raises(UnknownTickerError, match="ZZZZ, QQQQ"):
        resolve_tickers(parse_ticker_map(RAW), ["NVDA", "ZZZZ", "QQQQ"])


def test_fetch_then_load_round_trip(cache):
    fetcher = FakeFetcher({TICKER_MAP_URL: RAW})
    assert fetch_ticker_map(fetcher, cache).status == "downloaded"
    assert fetch_ticker_map(fetcher, cache).status == "cached"
    assert len(fetcher.calls) == 1
    assert "NVDA" in load_ticker_map(cache)["ticker"].tolist()


def test_load_without_cache_says_what_to_run(cache):
    with pytest.raises(FileNotFoundError, match="screener.data.pull"):
        load_ticker_map(cache)


def test_read_universe_dedupes_and_keeps_order(tmp_path):
    path = tmp_path / "u.csv"
    path.write_text("ticker\nnvda\nAMD\n\nNVDA\n# a comment\nbrk.b\n")
    assert read_universe(path) == ["NVDA", "AMD", "BRK-B"]


def test_read_universe_requires_ticker_column(tmp_path):
    path = tmp_path / "u.csv"
    path.write_text("symbol\nNVDA\n")
    with pytest.raises(ValueError):
        read_universe(path)
