"""Pull raw data for the universe into the local cache.

    python -m screener.data.pull                      # fetch whatever is missing
    python -m screener.data.pull --refresh prices     # re-download prices
    python -m screener.data.pull --refresh all        # re-download everything
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from screener import config
from screener.data.cache import Cache, FetchRecord, utc_now, write_manifest
from screener.data.edgar import fetch_companyfacts, fetch_submissions
from screener.data.http import SecClient
from screener.data.prices import PriceSource, YFinancePriceSource, fetch_prices
from screener.data.tickers import fetch_ticker_map, load_ticker_map, read_universe, split_known

REFRESH_CHOICES = ("tickers", "edgar", "prices", "all")


def _error(source: str, key: str, exc: Exception) -> FetchRecord:
    return FetchRecord(source, key, "", "", "error", 0, utc_now(), f"{type(exc).__name__}: {exc}")


def pull_universe(
    tickers: list[str],
    client: SecClient,
    cache: Cache,
    price_source: PriceSource | None,
    refresh: set[str],
    log=print,
) -> list[FetchRecord]:
    """Fetch everything for the given tickers; one failure does not stop the rest."""
    records = [fetch_ticker_map(client, cache, refresh="tickers" in refresh)]
    companies, missing = split_known(load_ticker_map(cache), tickers)
    for ticker in missing:
        log(f"[skip   ] {ticker:<6} not in the SEC ticker file (acquired, delisted or renamed?)")
        records.append(
            FetchRecord("edgar/company_tickers", ticker, "", "", "error", 0, utc_now(),
                        "ticker not in SEC company_tickers.json")
        )

    for i, row in enumerate(companies.itertuples(index=False), start=1):
        company: list[FetchRecord] = []
        try:
            company.append(fetch_companyfacts(client, cache, row.cik10, "edgar" in refresh))
        except Exception as exc:  # keep going; the failure is recorded and reported
            company.append(_error("edgar/companyfacts", row.cik10, exc))
        try:
            company.extend(fetch_submissions(client, cache, row.cik10, "edgar" in refresh))
        except Exception as exc:
            company.append(_error("edgar/submissions", row.cik10, exc))
        if price_source is not None:
            try:
                company.append(fetch_prices(price_source, cache, row.ticker, "prices" in refresh))
            except Exception as exc:
                company.append(_error(f"prices/{price_source.name}", row.ticker, exc))

        counts = Counter(r.status for r in company)
        summary = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
        log(f"[{i:>3}/{len(companies)}] {row.ticker:<6} CIK{row.cik10}  {summary}")
        records.extend(company)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--universe", type=Path, default=config.DEFAULT_UNIVERSE)
    parser.add_argument("--tickers", nargs="+", help="pull these tickers instead of the universe file")
    parser.add_argument("--refresh", nargs="+", choices=REFRESH_CHOICES, default=[],
                        help="re-download these even if cached")
    parser.add_argument("--skip-prices", action="store_true")
    args = parser.parse_args(argv)

    refresh = set(args.refresh)
    if "all" in refresh:
        refresh = {"tickers", "edgar", "prices"}

    try:
        client = SecClient(config.sec_user_agent())
    except config.ConfigError as exc:
        print(f"Configuration problem: {exc}", file=sys.stderr)
        return 2

    cache = Cache(config.data_dir())
    tickers = args.tickers or read_universe(args.universe)
    price_source = None if args.skip_prices else YFinancePriceSource()

    print(f"Pulling {len(tickers)} tickers into {cache.root}")
    records = pull_universe(tickers, client, cache, price_source, refresh)
    manifest = write_manifest(cache, records)

    totals = Counter(r.status for r in records)
    print("\nDone: " + ", ".join(f"{n} {status}" for status, n in sorted(totals.items())))
    print(f"Manifest: {manifest}")
    problems = [r for r in records if r.status in ("error", "not_found")]
    for r in problems:
        print(f"  {r.status.upper():<9} {r.source} {r.key}: {r.detail}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
