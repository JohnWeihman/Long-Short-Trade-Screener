"""Show the ranked pairs, or one pair in detail.

    python scripts/show_pairs.py                  # pairs that pass every filter
    python scripts/show_pairs.py --all            # every candidate, with why it failed
    python scripts/show_pairs.py NXPI MPWR        # one pair side by side (either order)

Reads data/pairs/ and data/scores/. Run `python -m screener.pairing.build` first.
"""

from __future__ import annotations

import argparse
import sys

from screener import config
from screener.cleaning.store import read_table
from screener.data.cache import Cache
from screener.pairing.build import AREA, SCORES_AREA
from screener.pairing.report import pair_detail, pairs_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="*", help="two tickers to show one pair")
    parser.add_argument("--all", action="store_true", help="include candidates that fail a filter")
    args = parser.parse_args(argv)

    cache = Cache(config.data_dir())
    try:
        pairs = read_table(cache, "pairs", AREA)
        scores = read_table(cache, "scores", SCORES_AREA)
        metrics = read_table(cache, "metrics", SCORES_AREA)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not args.tickers:
        print(f"As of {pairs['as_of'].iloc[0]:%Y-%m-%d}\n")
        print(pairs_table(pairs, include_failing=args.all))
        return 0
    if len(args.tickers) != 2:
        print("Give exactly two tickers to show a pair.", file=sys.stderr)
        return 1
    a, b = (t.upper() for t in args.tickers)
    match = pairs[((pairs["long"] == a) & (pairs["short"] == b)) | ((pairs["long"] == b) & (pairs["short"] == a))]
    if match.empty:
        print(f"{a} and {b} are not a candidate pair (they must be scored and in the same peer group).", file=sys.stderr)
        return 1
    print(pair_detail(match.iloc[0], scores, metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
