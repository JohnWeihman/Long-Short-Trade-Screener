"""Walk through one company's fundamentals line by line, so market cap,
enterprise value, EBITDA, free cash flow and ROIC can be checked by hand.

    python scripts/show_fundamentals.py NVDA

Reads data/scores/fundamentals.parquet. Run `python -m screener.scoring.build` first.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from screener import config
from screener.cleaning.store import read_table
from screener.data.cache import Cache
from screener.scoring.build import AREA, load_settings
from screener.scoring.fundamentals import position

QUARTER_LINES = ["revenue", "operating_income", "depreciation_amortization", "operating_cash_flow", "capex"]


def money(x) -> str:
    return "-" if x is None or pd.isna(x) else f"{x / 1e6:>14,.1f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ticker")
    args = parser.parse_args(argv)

    cache = Cache(config.data_dir())
    try:
        table = read_table(cache, "fundamentals", AREA)
        quarterly = read_table(cache, "metrics_quarterly")
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    rows = table[table["ticker"] == args.ticker.upper()]
    if rows.empty or pd.isna(rows.iloc[0].get("fiscal_year")):
        print(f"No fundamentals for {args.ticker.upper()}.", file=sys.stderr)
        return 1
    f = rows.iloc[0]
    tax = load_settings().tax_rate

    print(f"{f['ticker']}  peer group: {f['peer_group'] or '(unplaced)'}  as of {f['as_of']:%Y-%m-%d}  ({f['basis']} basis)")
    print(f"Latest quarter: FY{int(f['fiscal_year'])} Q{int(f['quarter'])}, ended {f['period_end']:%Y-%m-%d}, "
          f"filed by {f['known_on']:%Y-%m-%d}")
    print("All dollar amounts in $ millions. Shares and price are on today's share basis.\n")

    print("MARKET CAP")
    print(f"  shares ({f['shares_source']})          {f['shares'] / 1e6:>14,.1f} M")
    print(f"  x price on {f['price_date']:%Y-%m-%d}                    {f['price']:>14,.2f}")
    print(f"  = market cap                             {money(f['market_cap'])}")

    print("\nENTERPRISE VALUE")
    print(f"    short-term debt                        {money(f['short_term_debt'])}")
    print(f"  + long-term debt                         {money(f['long_term_debt'])}")
    print(f"  - cash                                   {money(f['cash'])}")
    print(f"  - short-term investments                 {money(f['short_term_investments'])}")
    print(f"  = net debt                               {money(f['net_debt'])}")
    print(f"  market cap + net debt = enterprise value {money(f['enterprise_value'])}")
    print("  (leases, preferred stock and minority interests are not included)")

    end = position(int(f["fiscal_year"]), int(f["quarter"]))
    q = quarterly[(quarterly["cik10"] == f["cik10"]) & quarterly["metric"].isin(QUARTER_LINES)]
    q = q.assign(pos=position(q["fiscal_year"].astype("int64"), q["quarter"].astype("int64")))
    q = q[(q["pos"] > end - 4) & (q["pos"] <= end)]
    labels = [f"FY{int(y)} Q{int(k)}" for y, k in
              q[["fiscal_year", "quarter"]].drop_duplicates().sort_values(["fiscal_year", "quarter"]).values]
    print("\nTRAILING TWELVE MONTHS  (* = quarter derived from year-to-date figures)")
    print(f"  {'':<28}" + "".join(f"{label:>13}" for label in labels) + f"{'TTM':>14}")
    for metric in QUARTER_LINES:
        line = q[q["metric"] == metric].sort_values("pos")
        cells = "".join(f"{v / 1e6:>12,.1f}{'*' if m != 'reported' else ' '}" for v, m in zip(line["val"], line["method"]))
        print(f"  {metric:<28}{cells}{money(f[f'{metric}_ttm'])}")

    print("\nEBITDA AND FREE CASH FLOW (TTM)")
    print(f"  operating income + D&A = EBITDA          {money(f['ebitda_ttm'])}")
    print(f"  operating cash flow - capex = FCF        {money(f['fcf_ttm'])}")
    if not pd.isna(f["ebitda_ttm"]) and f["ebitda_ttm"] > 0:
        print(f"  EV / EBITDA                              {f['enterprise_value'] / f['ebitda_ttm']:>14,.1f} x")

    print("\nROIC")
    print(f"  operating income TTM x (1 - {tax:.0%})         {money(f['nopat_ttm'])}")
    print(f"  equity + debt - cash - investments, now  {money(f['invested_capital'])}")
    print(f"  same, one year ago                       {money(f['invested_capital_1y_ago'])}")
    print(f"  ROIC = after-tax income / average        {f['roic']:>14.1%}" if not pd.isna(f["roic"]) else "  ROIC undefined")

    if f["notes"]:
        print(f"\nNotes: {f['notes']}")
    print(f"\nTo see the raw filings behind any line: python scripts/trace_metric.py {f['ticker']} <metric>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
