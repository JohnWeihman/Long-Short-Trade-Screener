"""Print the memo for a pair, list the memos, or show the API call log.

    python scripts/show_memo.py                   # list
    python scripts/show_memo.py TER AMAT          # the template memo (either order)
    python scripts/show_memo.py 1                 # by rank
    python scripts/show_memo.py TER AMAT --ai     # the model-written version, if there is one
    python scripts/show_memo.py --log             # API calls per run

Reads data/memos/. Run `python -m screener.memos.build` first.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from screener import config
from screener.cleaning.store import read_table, table_path
from screener.data.cache import Cache
from screener.memos.build import AREA


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("which", nargs="*", help="two tickers, or a rank number")
    parser.add_argument("--ai", action="store_true", help="show the model-written memo")
    parser.add_argument("--log", action="store_true", help="show the API call log")
    args = parser.parse_args(argv)
    cache = Cache(config.data_dir())
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.log:
        if not table_path(cache, "model_runs", AREA).exists():
            print("No runs logged yet.")
            return 0
        log = read_table(cache, "model_runs", AREA)
        columns = ["run_at", "as_of", "ai_model", "pairs", "ai_pairs", "cached", "generated", "fallbacks",
                   "api_calls", "input_tokens", "output_tokens", "estimated_cost_usd"]
        with pd.option_context("display.width", 200):
            print(log[columns].to_string(index=False))
        print(f"\nTotal API calls: {int(log['api_calls'].sum())}, estimated cost ${log['estimated_cost_usd'].sum():.4f}")
        return 0

    try:
        memos = read_table(cache, "memos", AREA)
    except FileNotFoundError:
        print("No memos yet. Run: python -m screener.memos.build", file=sys.stderr)
        return 1
    if not args.which:
        for memo in memos.itertuples():
            ai = f"AI {memo.ai_status}" + (f", written {memo.ai_written}" if memo.ai_written else "")
            print(f"{memo.rank:>2}. long {memo.long:<5} short {memo.short:<5} {memo.peer_group:<26} ({ai})")
        return 0
    if len(args.which) == 1 and args.which[0].isdigit():
        match = memos[memos["rank"] == int(args.which[0])]
    elif len(args.which) == 2:
        a, b = (t.upper() for t in args.which)
        match = memos[((memos["long"] == a) & (memos["short"] == b)) | ((memos["long"] == b) & (memos["short"] == a))]
    else:
        print("Give two tickers or one rank number.", file=sys.stderr)
        return 1
    if match.empty:
        print("No memo for that pair. Memos exist only for passing pairs.", file=sys.stderr)
        return 1
    row = match.iloc[0]
    if args.ai:
        if not isinstance(row["memo_ai"], str) or not row["memo_ai"]:
            print(f"No AI memo for this pair ({row['ai_status']}: {row['ai_note'] or 'no reason recorded'}).",
                  file=sys.stderr)
            return 1
        print(row["memo_ai"])
    else:
        print(row["memo"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
