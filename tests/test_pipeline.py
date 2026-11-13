import pandas as pd

from screener import pipeline
from screener.pipeline import FATAL, STEPS, append_run_log, check_prices, run, run_log_row

NAMES = ["data pull", "price check", "cleaning", "scoring", "deal check", "pairs", "memos"]


def recorder(calls, name, code=0, error=None):
    def main(args):
        calls.append((name, args))
        if error:
            raise error
        return code
    return main


def steps(calls, codes=None, error_at=None):
    codes = codes or {}
    out = []
    for name, _, with_refresh, without in STEPS:
        error = RuntimeError("boom") if name == error_at else None
        out.append((name, recorder(calls, name, codes.get(name, 0), error), with_refresh, without))
    return out


def test_every_step_runs_in_order_and_the_pull_refreshes():
    calls = []
    assert run(steps(calls), log=lambda _: None) == 0
    assert [name for name, _ in calls] == NAMES
    assert calls[0][1] == ["--refresh", "all"]


def test_no_refresh_skips_the_pull_and_the_price_check():
    calls, lines, results = [], [], []
    assert run(steps(calls), refresh=False, log=lines.append, results=results) == 0
    assert [name for name, _ in calls] == NAMES[2:]
    assert [r[:2] for r in results[:2]] == [("data pull", "skipped"), ("price check", "skipped")]


def test_a_step_with_problems_does_not_stop_the_run():
    calls = []
    assert run(steps(calls, codes={"data pull": 1}), log=lambda _: None) == 1
    assert len(calls) == len(NAMES)


def test_a_step_that_cannot_run_stops_the_rest():
    calls, lines = [], []
    assert run(steps(calls, codes={"scoring": 2}), log=lines.append) == FATAL
    assert [name for name, _ in calls] == NAMES[:4]
    assert any("not run: deal check, pairs, memos" in line for line in lines)

    calls, lines = [], []
    assert run(steps(calls, error_at="pairs"), log=lines.append) == FATAL
    assert any("pairs failed: RuntimeError: boom" in line for line in lines)


def write_prices(cache, ticker, last):
    path = cache.prices_path("yfinance", ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": pd.to_datetime(["2026-01-02", last]), "close": [1.0, 2.0]}).to_parquet(path, index=False)


def test_price_check(cache, monkeypatch):
    tickers = [f"T{i}" for i in range(20)]
    monkeypatch.setattr(pipeline, "read_universe", lambda _: tickers)
    today = pd.Timestamp("2026-11-16")  # a Monday
    for t in tickers:
        write_prices(cache, t, "2026-11-13")  # Friday: 3 days old is fine
    lines = []
    assert check_prices(cache=cache, today=today, log=lines.append) == 0
    assert "current for 20 of 20" in lines[0]
    write_prices(cache, "T0", "2026-11-01")  # one stale company out of 20 is tolerated
    assert check_prices(cache=cache, today=today, log=lines.append) == 0 and "stale: T0 (2026-11-01)" in lines[-1]
    for t in tickers[1:4]:
        write_prices(cache, t, "2026-11-01")  # 4 of 20 is too many
    assert check_prices(cache=cache, today=today, log=lines.append) == FATAL
    cache.prices_path("yfinance", "T5").unlink()
    assert check_prices(cache=cache, today=today, log=lines.append) == FATAL and "T5 (no prices)" in lines[-1]


def test_run_log(cache):
    started = pd.Timestamp("2026-11-16 23:00").to_pydatetime()
    results = [("data pull", "ok", 50.2), ("price check", "ok", 0.1), ("memos", "problems", 80.0)]
    row = run_log_row(started, results, refresh=True, published=False, cache=cache)
    assert (row["status"], row["refresh"], row["published"], row["as_of"]) == ("problems", True, False, None)
    assert row["steps"] == "data pull ok 50s; price check ok 0s; memos problems 80s"
    append_run_log(cache, row)
    log = append_run_log(cache, {**row, "status": "ok"})
    assert log["status"].tolist() == ["problems", "ok"]
