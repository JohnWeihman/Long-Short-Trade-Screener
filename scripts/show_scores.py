"""Show the ranked table for a peer group, or the detail behind one company's score.

    python scripts/show_scores.py                    # every peer group
    python scripts/show_scores.py analog_power       # one group
    python scripts/show_scores.py TXN                # one company, metric by metric

Reads data/scores/. Run `python -m screener.scoring.build` first.
"""

from __future__ import annotations

import argparse
import sys

from screener import config
from screener.cleaning.store import read_table
from screener.data.cache import Cache
from screener.scoring.build import AREA
from screener.scoring.report import company_detail, group_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", nargs="?", help="a peer group or a ticker (default: all groups)")
    args = parser.parse_args(argv)

    cache = Cache(config.data_dir())
    try:
        scores, metrics, ranks = (read_table(cache, n, AREA) for n in ("scores", "metrics", "metric_ranks"))
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    groups = sorted(scores["peer_group"].dropna().unique())
    as_of = scores["as_of"].iloc[0]
    print(f"As of {as_of:%Y-%m-%d}. Ranks: 0 = worst in group, 1 = best.\n")
    if args.name is None:
        print("\n\n".join(group_table(scores, metrics, g) for g in groups))
    elif args.name in groups:
        print(group_table(scores, metrics, args.name))
    elif args.name.upper() in set(scores["ticker"]):
        print(company_detail(args.name.upper(), scores, ranks))
        print(f"\nIngredients: python scripts/show_fundamentals.py {args.name.upper()}")
    else:
        print(f"'{args.name}' is neither a peer group ({', '.join(groups)}) nor a ticker in the scores.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
