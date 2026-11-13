"""The fact sheet for one pair: every number a memo may use, taken only
from tables the screener has already built (scores, metric ranks, pairs,
fundamentals). A memo writer, template or model, adds no numbers of its own.

The fact sheet is plain data (dicts, lists, strings, numbers) so it can be
stored as JSON, hashed, and handed to a model as-is.
"""

from __future__ import annotations

import math

import pandas as pd

PILLARS = ("valuation", "quality", "direction")

# metric -> (label, how to show its value)
DISPLAY = {
    "ebitda_yield": ("EV/EBITDA", "multiple_of_yield"),
    "ebit_yield": ("EV/EBIT", "multiple_of_yield"),
    "earnings_yield": ("P/E", "multiple_of_yield"),
    "sales_yield": ("EV/sales", "multiple_of_yield"),
    "fcf_yield": ("Free cash flow yield", "percent1"),
    "gross_margin": ("Gross margin", "percent"),
    "operating_margin": ("Operating margin", "percent"),
    "roic": ("ROIC", "percent"),
    "net_debt_to_ebitda": ("Net debt/EBITDA", "leverage"),
    "net_debt_to_ebit": ("Net debt/EBIT", "leverage"),
    "operating_margin_change": ("Operating margin change vs a year ago", "points"),
    "revenue_growth_acceleration": ("Revenue growth acceleration", "points"),
    "fcf_margin_change": ("Free cash flow margin change vs a year ago", "points"),
}


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or x is pd.NA or x is pd.NaT


def show(metric: str, value) -> str:
    """A metric value the way an analyst would write it."""
    if _missing(value):
        return "n/a"
    kind = DISPLAY[metric][1]
    if kind == "multiple_of_yield":
        return f"{1 / value:.1f}x" if value > 0 else "negative earnings"
    if kind == "percent":
        return f"{value:.0%}"
    if kind == "percent1":
        return f"{value:.1%}"
    if kind == "points":
        return f"{value * 100:+.1f} pts"
    if math.isinf(value):  # leverage
        return "debt with no earnings"
    return f"net cash ({value:.1f}x)" if value < 0 else f"{value:.1f}x"


def _num(x, digits: int = 2):
    return None if _missing(x) else round(float(x), digits)


def _comparisons(ranks: pd.DataFrame, long: str, short: str) -> list[dict]:
    """Every metric slot for the two companies: values, ranks, and who is ahead."""
    by = ranks[ranks["ticker"].isin([long, short])].set_index(["metric", "ticker"])
    rows = []
    for metric in dict.fromkeys(ranks["metric"]):
        if (metric, long) not in by.index or (metric, short) not in by.index:
            continue
        a, b = by.loc[(metric, long)], by.loc[(metric, short)]
        if _missing(a["rank"]) or _missing(b["rank"]):
            continue
        used_a = a["used_metric"] if not _missing(a["used_metric"]) else metric
        used_b = b["used_metric"] if not _missing(b["used_metric"]) else metric
        label = DISPLAY[used_a][0] if used_a == used_b else f"{DISPLAY[used_a][0]} / {DISPLAY[used_b][0]}"
        rows.append({
            "metric": metric, "pillar": a["pillar"], "label": label,
            "long": show(used_a, a["value"]), "short": show(used_b, b["value"]),
            "rank_long": _num(a["rank"]), "rank_short": _num(b["rank"]),
            "edge": _num(a["rank"] - b["rank"]),
        })
    return rows


def fact_sheet(
    pair: pd.Series, scores: pd.DataFrame, metric_ranks: pd.DataFrame, fundamentals: pd.DataFrame,
    deals: dict[str, dict] | None = None, n_passing: int | None = None,
) -> dict:
    """Everything known about one pair. deals maps ticker -> {'status', 'evidence'}."""
    long, short = pair["long"], pair["short"]
    s, f = scores.set_index("ticker"), fundamentals.set_index("ticker")
    deals = deals or {}

    def company(ticker: str) -> dict:
        row, fund = s.loc[ticker], f.loc[ticker]
        deal = deals.get(ticker, {})
        return {
            "ticker": ticker,
            "composite": _num(row["composite"]),
            "rank_in_group": None if _missing(row["rank_in_group"]) else int(row["rank_in_group"]),
            "group_size": None if _missing(row["group_size"]) else int(row["group_size"]),
            "pillars": {p: _num(row[p]) for p in PILLARS},
            "latest_quarter": f"FY{int(fund['fiscal_year'])} Q{int(fund['quarter'])}",
            "quarter_ended": f"{fund['period_end']:%Y-%m-%d}",
            "price": _num(fund["price"]),
            "market_cap_bn": _num(fund["market_cap"] / 1e9, 1),
            "revenue_ttm_bn": _num(fund["revenue_ttm"] / 1e9, 2),
            # Routine notes ("X not reported: taken as 0") stay in the fundamentals table.
            "data_notes": "; ".join(n for n in (fund["notes"] or "").split("; ") if n and "taken as 0" not in n),
            "deal_status": deal.get("status", "none"),
            "deal_evidence": deal.get("evidence", "") if deal.get("status", "none") != "none" else "",
        }

    comparisons = _comparisons(metric_ranks, long, short)
    ordered = sorted(comparisons, key=lambda c: c["edge"], reverse=True)
    return {
        "pair_id": pair["pair_id"], "long": long, "short": short, "peer_group": pair["peer_group"],
        "as_of": f"{pd.Timestamp(pair['as_of']):%Y-%m-%d}",
        "rank": None if _missing(pair["rank"]) else int(pair["rank"]),
        "passing_pairs": n_passing,
        "thin_group": bool(pair["thin_group"]),
        "composite_gap": _num(pair["gap"]),
        "pillar_gaps": {p: _num(pair[f"gap_{p}"]) for p in PILLARS},
        "companies": {"long": company(long), "short": company(short)},
        "comparisons": comparisons,
        "for_the_trade": [c for c in ordered if c["edge"] > 0][:4],
        "against_the_trade": [c for c in reversed(ordered) if c["edge"] < 0][:3],
        "trade": {
            "correlation": _num(pair["correlation"]), "weeks": int(pair["overlap_weeks"]),
            "beta_long": _num(pair["beta_long"]), "beta_short": _num(pair["beta_short"]),
            "hedge_ratio": _num(pair["hedge_ratio"]), "share_ratio": _num(pair["share_ratio"]),
            "spread_z": _num(pair["spread_z"], 1),
            "spread_chg_1m": _num(pair["spread_chg_1m"], 3), "spread_chg_3m": _num(pair["spread_chg_3m"], 3),
            "spread_chg_12m": _num(pair["spread_chg_12m"], 3),
        },
    }
