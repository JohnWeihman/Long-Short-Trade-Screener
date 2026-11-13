"""Refresh the data and rebuild everything, in order, with one command.

    python -m screener.pipeline                 # refresh EDGAR + prices, rebuild all tables and memos
    python -m screener.pipeline --no-refresh    # rebuild from the local cache only (no downloads)

Steps: data pull -> cleaning -> scoring -> deal check -> pairs -> memos.
A pull refresh re-downloads the ticker map, companyfacts, submissions and
prices (about 3 SEC requests per company); filing documents are fetched only
when a filing is new. This is the command a scheduled job should run.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

from screener.cleaning import build as cleaning_build
from screener.data import pull
from screener.memos import build as memos_build
from screener.pairing import build as pairing_build
from screener.pairing import deals
from screener.scoring import build as scoring_build

# (name, the step's main function, arguments with refresh, arguments without)
Step = tuple[str, Callable[[list[str]], int], list[str], list[str]]
STEPS: list[Step] = [
    ("data pull", pull.main, ["--refresh", "all"], None),  # None: skipped with --no-refresh
    ("cleaning", cleaning_build.main, [], []),
    ("scoring", scoring_build.main, [], []),
    ("deal check", deals.main, [], []),
    ("pairs", pairing_build.main, [], []),
    ("memos", memos_build.main, [], []),
]
# Exit code 1 from a step means "finished, with some problems" (for example a
# ticker the SEC no longer lists); anything higher means it could not run.
FATAL = 2


def run(steps: list[Step], refresh: bool = True, log=print) -> int:
    """Run the steps in order. Stops at the first step that cannot run; a
    step that finishes with problems is reported and the next one runs."""
    results = []
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
    if any(status == "FAILED" for _, status, _ in results):
        return FATAL
    return 1 if any(status == "problems" for _, status, _ in results) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-refresh", action="store_true", help="skip the data pull; rebuild from the cache")
    args = parser.parse_args(argv)
    return run(STEPS, refresh=not args.no_refresh)


if __name__ == "__main__":
    raise SystemExit(main())
