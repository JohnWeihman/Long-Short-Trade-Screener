"""Refresh the data and rebuild everything, in order, with one command.

    python -m screener.pipeline                 # refresh EDGAR + prices, rebuild all tables and memos
    python -m screener.pipeline --no-refresh    # rebuild from the local cache only (no downloads)
    python -m screener.pipeline --publish       # also write published/ for the web app (the scheduled job)

Steps: data pull -> cleaning -> scoring -> deal check -> pairs -> memos
[-> publish]. Before the steps, run state published by an earlier run (the
AI memo cache and the run logs) is restored into data/ where data/ lacks it,
so a fresh machine does not pay for AI memos again. A pull refresh
re-downloads the ticker map, companyfacts, submissions and prices (about 3
SEC requests per company); filing documents are fetched only when new. Every
run is appended to data/logs/pipeline_runs.parquet.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from screener import config, publish
from screener.cleaning import build as cleaning_build
from screener.cleaning.store import read_table, table_path, write_table
from screener.data import pull
from screener.data.cache import Cache
from screener.data.tickers import read_universe
from screener.memos import build as memos_build
from screener.pairing import build as pairing_build
from screener.pairing import deals
from screener.scoring import build as scoring_build

MAX_PRICE_AGE_DAYS = 5  # covers a weekend plus a holiday
MAX_STALE_SHARE = 0.10


def check_prices(argv: list[str] | None = None, cache: Cache | None = None, today: pd.Timestamp | None = None,
                 log=print) -> int:
    """Fail (exit 2) when the price refresh did not really happen: more than
    10% of the universe without a price in the last 5 days. Yahoo sometimes
    blocks cloud servers; publishing yesterday's results is better than
    publishing results built on stale prices."""
    cache = cache or Cache(config.data_dir())
    today = (today or pd.Timestamp.today()).normalize()
    tickers = read_universe(config.DEFAULT_UNIVERSE)
    stale = []
    for ticker in tickers:
        path = cache.prices_path("yfinance", ticker)
        latest = pd.read_parquet(path, columns=["date"])["date"].max() if path.exists() else None
        if latest is None or (today - pd.Timestamp(latest).normalize()).days > MAX_PRICE_AGE_DAYS:
            stale.append(f"{ticker} ({'no prices' if latest is None else f'{pd.Timestamp(latest):%Y-%m-%d}'})")
    if len(stale) > MAX_STALE_SHARE * len(tickers):
        log(f"Prices are stale for {len(stale)} of {len(tickers)} companies: {', '.join(stale[:10])}")
        return FATAL
    log(f"Prices are current for {len(tickers) - len(stale)} of {len(tickers)} companies"
        + (f"; stale: {', '.join(stale)}" if stale else "") + ".")
    return 0


# (name, the step's main function, arguments with refresh, arguments without)
Step = tuple[str, Callable[[list[str]], int], list[str], list[str] | None]
STEPS: list[Step] = [
    ("data pull", pull.main, ["--refresh", "all"], None),  # None: skipped with --no-refresh
    ("price check", check_prices, [], None),
    ("cleaning", cleaning_build.main, [], []),
    ("scoring", scoring_build.main, [], []),
    ("deal check", deals.main, [], []),
    ("pairs", pairing_build.main, [], []),
    ("memos", memos_build.main, [], []),
]
PUBLISH_STEP: Step = ("publish", publish.main, [], [])
# Exit code 1 from a step means "finished, with some problems" (for example a
# ticker the SEC no longer lists); anything higher means it could not run.
FATAL = 2
RUN_LOG_COLUMNS = ["started_at", "finished_at", "status", "refresh", "published", "steps", "as_of", "passing_pairs"]


def run(steps: list[Step], refresh: bool = True, log=print, results: list | None = None) -> int:
    """Run the steps in order. Stops at the first step that cannot run; a
    step that finishes with problems is reported and the next one runs.
    Each step's (name, status, seconds) is appended to `results` if given."""
    results = [] if results is None else results
    for name, main, with_refresh, without in steps:
        args = with_refresh if refresh else without
        if args is None:
            results.append((name, "skipped", 0.0))
            continue
        log(f"\n===== {name} =====")
        started = time.monotonic()
        try:
            code = main(args) or 0
        except Exception as exc:  # report and stop: later steps depend on this one
            log(f"{name} failed: {type(exc).__name__}: {exc}")
            code = FATAL
        elapsed = time.monotonic() - started
        results.append((name, "ok" if code == 0 else ("problems" if code < FATAL else "FAILED"), elapsed))
        if code >= FATAL:
            break

    log("\n===== pipeline summary =====")
    for name, status, elapsed in results:
        log(f"  {name:<11} {status:<9} {elapsed:6.1f}s")
    not_run = [name for name, *_ in steps[len(results):]]
    if not_run:
        log(f"  not run: {', '.join(not_run)}")
    return overall_status(results)


def overall_status(results: list) -> int:
    if any(status == "FAILED" for _, status, _ in results):
        return FATAL
    return 1 if any(status == "problems" for _, status, _ in results) else 0


def run_log_row(started: datetime, results: list, refresh: bool, published: bool, cache: Cache) -> dict:
    as_of, passing = None, None
    if table_path(cache, "pairs", "pairs").exists():
        pairs = read_table(cache, "pairs", "pairs")
        if len(pairs):
            as_of, passing = f"{pd.Timestamp(pairs['as_of'].iloc[0]):%Y-%m-%d}", int(pairs["passes"].sum())
    code = overall_status(results)
    return {
        "started_at": started.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "status": {0: "ok", 1: "problems"}.get(code, "FAILED"), "refresh": refresh, "published": published,
        "steps": "; ".join(f"{name} {status} {seconds:.0f}s" for name, status, seconds in results),
        "as_of": as_of, "passing_pairs": passing,
    }


def append_run_log(cache: Cache, row: dict) -> pd.DataFrame:
    new = pd.DataFrame([row], columns=RUN_LOG_COLUMNS)
    path = table_path(cache, "pipeline_runs", "logs")
    log = pd.concat([read_table(cache, "pipeline_runs", "logs"), new], ignore_index=True) if path.exists() else new
    write_table(cache, "pipeline_runs", log, "logs")
    return log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-refresh", action="store_true", help="skip the data pull; rebuild from the cache")
    parser.add_argument("--publish", action="store_true", help="write published/ for the web app at the end")
    args = parser.parse_args(argv)
    cache = Cache(config.data_dir())
    started = datetime.now(timezone.utc)

    restored = publish.restore(cache)
    if restored:
        print("Restored from published/: " + ", ".join(restored))
    results: list = []
    steps = STEPS + ([PUBLISH_STEP] if args.publish else [])
    code = run(steps, refresh=not args.no_refresh, results=results)
    published = any(name == "publish" and status == "ok" for name, status, _ in results)
    append_run_log(cache, run_log_row(started, results, not args.no_refresh, published, cache))
    if published:
        # Publish once more so published/ includes this run's own log line.
        publish.publish(cache)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
