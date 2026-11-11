"""Plain-text views of the scores, shared by the build command and scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _multiple(x) -> str:
    return "-" if pd.isna(x) else f"{x:,.1f}x"


def _percent(x) -> str:
    return "-" if pd.isna(x) else f"{x:.0%}"


def _points(x) -> str:
    return "-" if pd.isna(x) else f"{x * 100:+.1f}"


def _score(x) -> str:
    return "  - " if pd.isna(x) else f"{x:.2f}"


def _leverage(x) -> str:
    if pd.isna(x):
        return "-"
    return "no earn." if np.isinf(x) else f"{x:,.1f}x"


def group_table(scores: pd.DataFrame, metrics: pd.DataFrame, group: str) -> str:
    """A peer group ranked by composite score, with the ratios behind it."""
    rows = scores[scores["peer_group"] == group].merge(
        metrics.drop(columns=["cik10", "peer_group", "as_of"]), on="ticker", how="left"
    ).sort_values(["rank_in_group", "ticker"], na_position="last")
    thin = " (thin group: fewer companies than the minimum, treat ranks with caution)" if rows["thin_group"].any() else ""
    lines = [
        f"{group} - {len(rows)} companies{thin}",
        f"{'#':>2} {'ticker':<6} {'score':>5}  {'val':>4} {'qual':>4} {'dir':>4} |"
        f"{'EV/EBITDA':>10} {'EV/EBIT':>8} {'P/E':>7} {'FCF yld':>8} {'EV/S':>6} |"
        f"{'GM':>5} {'OM':>5} {'ROIC':>6} {'ND/EBITDA':>10} |"
        f"{'dOM':>6} {'accel':>7} {'dFCFm':>6}",
    ]
    for r in rows.itertuples():
        place = "-" if pd.isna(r.rank_in_group) else str(int(r.rank_in_group))
        debt = r.net_debt_to_ebitda if not pd.isna(r.net_debt_to_ebitda) else r.net_debt_to_ebit
        lines.append(
            f"{place:>2} {r.ticker:<6} {_score(r.composite):>5}  {_score(r.valuation):>4} {_score(r.quality):>4} "
            f"{_score(r.direction):>4} |"
            f"{_multiple(r.ev_ebitda):>10} {_multiple(r.ev_ebit):>8} {_multiple(r.pe):>7} {_percent(r.fcf_yield):>8} "
            f"{_multiple(r.ev_sales):>6} |"
            f"{_percent(r.gross_margin):>5} {_percent(r.operating_margin):>5} {_percent(r.roic):>6} {_leverage(debt):>10} |"
            f"{_points(r.operating_margin_change):>6} {_points(r.revenue_growth_acceleration):>7} "
            f"{_points(r.fcf_margin_change):>6}"
        )
    return "\n".join(lines)


def company_detail(ticker: str, scores: pd.DataFrame, metric_ranks: pd.DataFrame) -> str:
    """Every metric behind one company's score: value, rank in group, metric used."""
    row = scores[scores["ticker"] == ticker]
    if row.empty:
        return f"{ticker} is not in the scores table."
    s = row.iloc[0]
    if s["peer_group"] is None or pd.isna(s["peer_group"]):
        return f"{ticker} has no peer group, so it is not ranked."
    place = "-" if pd.isna(s["rank_in_group"]) else f"{int(s['rank_in_group'])} of {int(s['group_size'])}"
    lines = [
        f"{ticker} - {s['peer_group']} - rank {place} - composite {_score(s['composite'])}",
        "Ranks run from 0 (worst in the group) to 1 (best).",
    ]
    ranks = metric_ranks[metric_ranks["ticker"] == ticker]
    for pillar, group in ranks.groupby("pillar", sort=False):
        lines.append(f"\n{pillar}: {_score(s[pillar])}")
        for r in group.itertuples():
            value = "-" if pd.isna(r.value) else ("no earnings" if np.isinf(r.value) else f"{r.value:,.3f}")
            note = "" if r.used_metric in (None, r.metric) or pd.isna(r.used_metric) else f"  (used {r.used_metric})"
            lines.append(f"  {r.metric:<30} value {value:>12}   rank {_score(r.rank)}{note}")
    return "\n".join(lines)
