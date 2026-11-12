"""Plain-text views of the pairs, shared by the build command and scripts."""

from __future__ import annotations

import pandas as pd


def _num(x, spec: str = ".2f") -> str:
    return "-" if x is None or pd.isna(x) else format(x, spec)


def _pct(x) -> str:
    return "-" if x is None or pd.isna(x) else f"{x:+.0%}"


def _short_notes(notes: str) -> str:
    """The notes that matter at a glance. Routine ones ("X not reported:
    taken as 0") are left for the pair detail view."""
    if not notes:
        return ""
    kept = []
    for part in notes.split(" | "):
        ticker, _, text = part.partition(": ")
        if not text:  # a pair-level note such as "thin peer group"
            kept.append(part)
            continue
        important = [item for item in text.split("; ") if "taken as 0" not in item]
        if important:
            kept.append(f"{ticker}: {'; '.join(important)}")
    return " | ".join(kept)


def pairs_table(pairs: pd.DataFrame, include_failing: bool = False) -> str:
    """Ranked pairs, one per line. Failing candidates (with the reason) are
    listed after the passing ones when include_failing is set."""
    shown = pairs if include_failing else pairs[pairs["passes"]]
    if shown.empty:
        return "No pairs pass the filters."
    lines = [
        f"{'#':>3} {'long':<5} {'short':<5} {'group':<26} {'gap':>5} {'val':>5} {'qual':>5} {'dir':>5} |"
        f"{'corr':>5} {'hedge':>6} {'sh/sh':>6} |{'z':>5} {'3m':>5} {'12m':>6}  notes",
    ]
    for r in shown.itertuples():
        place = "-" if pd.isna(r.rank) else str(int(r.rank))
        tail = _short_notes(r.notes) if r.passes else f"FAILS: {r.reasons}"
        lines.append(
            f"{place:>3} {r.long:<5} {r.short:<5} {r.peer_group:<26} {_num(r.gap):>5} {_num(r.gap_valuation, '+.2f'):>5} "
            f"{_num(r.gap_quality, '+.2f'):>5} {_num(r.gap_direction, '+.2f'):>5} |"
            f"{_num(r.correlation):>5} {_num(r.hedge_ratio):>6} {_num(r.share_ratio):>6} |"
            f"{_num(r.spread_z, '+.1f'):>5} {_pct(r.spread_chg_3m):>5} {_pct(r.spread_chg_12m):>6}  {tail}"
        )
    lines.append(
        "\ngap/val/qual/dir: long's score minus short's (0-1 scale). corr: weekly return correlation. "
        "hedge: $ short per $1 long.\nsh/sh: shares short per share long. z: where the hedged spread is "
        "against its last year. 3m/12m: spread move."
    )
    return "\n".join(lines)


def pair_detail(pair: pd.Series, scores: pd.DataFrame, metrics: pd.DataFrame) -> str:
    """One pair side by side: scores, key ratios, and the hedge arithmetic."""
    long, short = pair["long"], pair["short"]
    s = scores.set_index("ticker")
    m = metrics.set_index("ticker")
    status = f"rank {int(pair['rank'])}" if pair["passes"] else f"does not qualify: {pair['reasons']}"
    lines = [
        f"LONG {long} / SHORT {short} - {pair['peer_group']} - {status}",
        f"As of {pair['as_of']:%Y-%m-%d}\n",
        f"{'':<30}{long:>10}{short:>10}{'gap':>9}",
    ]
    for label, column in (("composite score", "composite"), ("valuation", "valuation"),
                          ("quality", "quality"), ("direction", "direction")):
        a, b = s.loc[long, column], s.loc[short, column]
        lines.append(f"{label:<30}{_num(a):>10}{_num(b):>10}{_num(a - b, '+.2f'):>9}")
    lines.append("")
    ratios = (
        ("EV/EBITDA", "ev_ebitda", ".1f"), ("EV/EBIT", "ev_ebit", ".1f"), ("P/E", "pe", ".1f"),
        ("FCF yield", "fcf_yield", ".1%"), ("EV/sales", "ev_sales", ".1f"),
        ("gross margin", "gross_margin", ".0%"), ("operating margin", "operating_margin", ".0%"),
        ("ROIC", "roic", ".0%"), ("net debt/EBITDA", "net_debt_to_ebitda", ".1f"),
        ("revenue growth (TTM)", "revenue_growth", "+.0%"),
        ("operating margin change", "operating_margin_change", "+.1%"),
    )
    for label, column, spec in ratios:
        lines.append(f"{label:<30}{_num(m.loc[long, column], spec):>10}{_num(m.loc[short, column], spec):>10}")

    lines += [
        "",
        f"Weekly return correlation       {_num(pair['correlation'])}  over {int(pair['overlap_weeks'])} weeks "
        f"(with the sector move removed: {_num(pair['residual_correlation'])})",
        f"Beta to the universe index      {long} {_num(pair['beta_long'])}, {short} {_num(pair['beta_short'])}",
        f"Hedge ratio                     {_num(pair['beta_long'])} / {_num(pair['beta_short'])} = "
        f"{_num(pair['hedge_ratio'])}  -> short ${_num(pair['hedge_ratio'])} of {short} per $1.00 of {long}",
        f"In shares                       {_num(pair['hedge_ratio'])} x {_num(pair['price_long'])} / "
        f"{_num(pair['price_short'])} = {_num(pair['share_ratio'])} {short} shares per {long} share",
        f"Direct pair beta                {_num(pair['pair_beta'])}  ({long} returns regressed on {short} returns)",
        "",
        f"Hedged spread: z-score {_num(pair['spread_z'], '+.1f')} against the last year; moved "
        f"{_pct(pair['spread_chg_1m'])} in 1 month, {_pct(pair['spread_chg_3m'])} in 3 months, "
        f"{_pct(pair['spread_chg_12m'])} in 12 months.",
        "A positive move means the long has already outperformed the hedged short.",
    ]
    if pair["notes"]:
        lines.append(f"\nNotes: {pair['notes']}")
    return "\n".join(lines)
