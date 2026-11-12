"""Candidate pairs: every two scored companies in the same peer group,
long the higher composite score and short the lower.

Nothing is dropped here. Each candidate records whether it clears the
score-gap and pillar-agreement rules, so the full list stays visible.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from screener.pairing.config import PairingConfig

PILLARS = ("valuation", "quality", "direction")
COLUMNS = [
    "pair_id", "peer_group", "long", "short", "composite_long", "composite_short", "gap",
    "gap_valuation", "gap_quality", "gap_direction", "pillars_favoring_long", "thin_group",
    "passes_gap", "passes_pillars",
]


def candidate_pairs(scores: pd.DataFrame, cfg: PairingConfig) -> pd.DataFrame:
    """One row per pair of scored companies within a peer group."""
    scored = scores[scores["peer_group"].notna() & scores["composite"].notna()]
    rows = []
    for group, members in scored.groupby("peer_group", sort=True):
        by_ticker = members.set_index("ticker")
        for a, b in combinations(sorted(by_ticker.index), 2):
            # Higher composite is the long; on an exact tie the first ticker alphabetically.
            long, short = (a, b) if by_ticker.loc[a, "composite"] >= by_ticker.loc[b, "composite"] else (b, a)
            top, bottom = by_ticker.loc[long], by_ticker.loc[short]
            gaps = {p: top[p] - bottom[p] for p in PILLARS if p in by_ticker.columns}
            favoring = sum(1 for g in gaps.values() if pd.notna(g) and g > 0)
            gap = float(top["composite"] - bottom["composite"])
            rows.append({
                "pair_id": f"{long}/{short}", "peer_group": group, "long": long, "short": short,
                "composite_long": float(top["composite"]), "composite_short": float(bottom["composite"]), "gap": gap,
                **{f"gap_{p}": (float(g) if pd.notna(g) else None) for p, g in gaps.items()},
                "pillars_favoring_long": favoring,
                "thin_group": bool(top.get("thin_group", False)),
                # Tolerance so a gap that is exactly the threshold is not lost to float rounding.
                "passes_gap": gap >= cfg.min_composite_gap - 1e-9,
                "passes_pillars": favoring >= cfg.min_pillars_favoring_long,
            })
    frame = pd.DataFrame(rows, columns=COLUMNS)
    return frame.sort_values(["gap", "pair_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
