"""Print the memo for a pair, or list the memos available.

    python scripts/show_memo.py                   # list
    python scripts/show_memo.py TER AMAT          # one memo (either order)
    python scripts/show_memo.py 1                 # by rank

Reads data/memos/. Run `python -m screener.memos.build` first.
"""

from __future__ import annotations

import argparse
import sys

from screener import config
from screener.cleaning.store import read_table
from screener.data.cache import Cache
from screener.memos.build import AREA


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("which", nargs="*", help="two tickers, or a rank number")
    args = parser.parse_args(argv)
    try:
        memos = read_table(Cache(config.data_dir()), "memos", AREA)
    except FileNotFoundError:
        print("No memos yet. Run: python -m screener.memos.build", file=sys.stderr)
        return 1

    if not args.which:
        for memo in memos.itertuples():
            print(f"{memo.rank:>2}. long {memo.long:<5} short {memo.short:<5} {memo.peer_group:<26} ({memo.writer})")
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
        print("No memo for that pair. Memos exist only for the top-ranked passing pairs.", file=sys.stderr)
        return 1
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(match.iloc[0]["memo"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
