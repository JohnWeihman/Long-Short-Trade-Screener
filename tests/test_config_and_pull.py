import pandas as pd
import pytest

from screener import config
from screener.data.prices import PriceDataError, normalize_yfinance
from screener.data.pull import main, pull_universe
from screener.data.tickers import TICKER_MAP_URL

from conftest import FakeFetcher

TICKERS = {
    "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "1": {"cik_str": 2488, "ticker": "AMD", "title": "ADVANCED MICRO DEVICES INC"},
}


def edgar_responses(ciks):
    out = {TICKER_MAP_URL: TICKERS}
    for cik in ciks:
        out[f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"] = {"cik": int(cik)}
        out[f"https://data.sec.gov/submissions/CIK{cik}.json"] = {"filings": {"files": []}}
    return out


class FakePrices:
    name = "fake"

    def __init__(self, fail=()):
        self.fail = set(fail)

    def history(self, ticker):
        if ticker in self.fail:
            raise PriceDataError(f"No price data returned for {ticker}")
        index = pd.DatetimeIndex(["2024-06-10"], tz="America/New_York")
        raw = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1]}, index=index)
        return normalize_yfinance(raw, ticker)


def test_user_agent_must_be_set(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(config.ConfigError, match="not set"):
        config.sec_user_agent()


@pytest.mark.parametrize("value", ["Your Name your.email@example.com", "No Email Here"])
def test_user_agent_placeholder_is_rejected(monkeypatch, value):
    monkeypatch.setenv("SEC_USER_AGENT", value)
    with pytest.raises(config.ConfigError, match="placeholder"):
        config.sec_user_agent()


def test_user_agent_real_value_is_returned(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "  Jane Doe jane@fund.test ")
    assert config.sec_user_agent() == "Jane Doe jane@fund.test"


def test_data_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SCREENER_DATA_DIR", str(tmp_path))
    assert config.data_dir() == tmp_path
    monkeypatch.delenv("SCREENER_DATA_DIR")
    assert config.data_dir() == config.PROJECT_ROOT / "data"


def test_main_stops_before_any_request_without_a_user_agent(monkeypatch, capsys):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    assert main(["--tickers", "NVDA"]) == 2
    assert "SEC_USER_AGENT" in capsys.readouterr().err


def test_pull_fetches_everything_then_nothing_on_rerun(cache):
    fetcher = FakeFetcher(edgar_responses(["0001045810", "0000002488"]))
    first = pull_universe(["NVDA", "AMD"], fetcher, cache, FakePrices(), set(), log=lambda _: None)
    assert [r.status for r in first] == ["downloaded"] * 7  # ticker map + 3 per company
    calls_after_first = len(fetcher.calls)

    second = pull_universe(["NVDA", "AMD"], fetcher, cache, FakePrices(), set(), log=lambda _: None)
    assert [r.status for r in second] == ["cached"] * 7
    assert len(fetcher.calls) == calls_after_first


def test_refresh_prices_leaves_edgar_cache_alone(cache):
    fetcher = FakeFetcher(edgar_responses(["0001045810"]))
    pull_universe(["NVDA"], fetcher, cache, FakePrices(), set(), log=lambda _: None)
    calls = len(fetcher.calls)
    records = pull_universe(["NVDA"], fetcher, cache, FakePrices(), {"prices"}, log=lambda _: None)
    assert {r.source: r.status for r in records} == {
        "edgar/company_tickers": "cached",
        "edgar/companyfacts": "cached",
        "edgar/submissions": "cached",
        "prices/fake": "downloaded",
    }
    assert len(fetcher.calls) == calls


def test_unknown_ticker_is_reported_and_the_rest_still_pulled(cache):
    fetcher = FakeFetcher(edgar_responses(["0001045810"]))
    lines = []
    records = pull_universe(["GONE", "NVDA"], fetcher, cache, FakePrices(), set(), log=lines.append)
    gone = [r for r in records if r.key == "GONE"]
    assert [(r.status, r.detail) for r in gone] == [("error", "ticker not in SEC company_tickers.json")]
    assert any("GONE" in line for line in lines)
    assert cache.companyfacts_path(1045810).exists()


def test_one_failure_does_not_stop_the_rest(cache):
    responses = edgar_responses(["0000002488"])  # NVDA's EDGAR files are missing
    fetcher = FakeFetcher(responses)
    records = pull_universe(["NVDA", "AMD"], fetcher, cache, FakePrices(fail={"NVDA"}), set(), log=lambda _: None)
    by_key = {(r.source, r.key): r.status for r in records}
    assert by_key[("edgar/companyfacts", "0001045810")] == "not_found"
    assert by_key[("prices/fake", "NVDA")] == "error"
    assert by_key[("edgar/companyfacts", "0000002488")] == "downloaded"
    assert by_key[("prices/fake", "AMD")] == "downloaded"
