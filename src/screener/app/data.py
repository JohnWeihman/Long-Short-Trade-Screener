"""Loading the published tables and shaping them for the app's pages.

The app reads only published/ (committed, refreshed by the scheduled job).
It never touches data/, the SEC or the Anthropic API. Everything here is a
plain function of DataFrames, so it is tested without running Streamlit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from screener import config

QUARTER_LINES = {
    "revenue_growth": "Revenue growth vs year-ago quarter",
    "gross_margin": "Gross margin",
    "operating_margin": "Operating margin",
    "fcf_margin_ttm": "Free cash flow margin (TTM)",
}
DEAL_LABELS = {
    "pending_target": "being acquired",
    "pending_other": "pending deal",
    "closed": "recent deal",
    "none": "",
}


def published_dir() -> Path:
    """published/ in the project, or SCREENER_PUBLISHED_DIR if set."""
    override = (os.environ.get("SCREENER_PUBLISHED_DIR") or "").strip()
    return Path(override) if override else config.PROJECT_ROOT / "published"


def load_table(folder: Path, name: str) -> pd.DataFrame | None:
    path = Path(folder) / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def load_manifest(folder: Path) -> dict | None:
    path = Path(folder) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---- pairs ---------------------------------------------------------------

def deal_flag(long_status: str, short_status: str, long: str, short: str) -> str:
    parts = [f"{t}: {DEAL_LABELS.get(s, s)}" for t, s in ((long, long_status), (short, short_status))
             if DEAL_LABELS.get(s, s)]
    return "; ".join(parts)


def pairs_view(pairs: pd.DataFrame, include_failing: bool = False, groups: list[str] | None = None) -> pd.DataFrame:
    """The ranked pair list as shown on the Pairs page. Percent figures are
    multiplied by 100 so the table can format them."""
    rows = pairs if include_failing else pairs[pairs["passes"]]
    if groups:
        rows = rows[rows["peer_group"].isin(groups)]
    rows = rows.sort_values(["rank", "gap"], ascending=[True, False], na_position="last")
    out = pd.DataFrame({
        "pair_id": rows["pair_id"],
        "Rank": rows["rank"],
        "Long": rows["long"],
        "Short": rows["short"],
        "Peer group": rows["peer_group"],
        "Score gap": rows["gap"],
        "Valuation gap": rows["gap_valuation"],
        "Quality gap": rows["gap_quality"],
        "Direction gap": rows["gap_direction"],
        "Correlation": rows["correlation"],
        "Hedge ratio": rows["hedge_ratio"],
        "Spread z": rows["spread_z"],
        "Spread 3m %": rows["spread_chg_3m"] * 100,
        "Spread 12m %": rows["spread_chg_12m"] * 100,
        "Deals": [deal_flag(a, b, l, s) for a, b, l, s in
                  zip(rows.get("deal_long", "none"), rows.get("deal_short", "none"), rows["long"], rows["short"])],
        "Status": ["passes" if p else f"fails: {r}" for p, r in zip(rows["passes"], rows["reasons"].fillna(""))],
    })
    return out.reset_index(drop=True)


def pair_label(row) -> str:
    rank = "" if pd.isna(row["rank"]) else f"{int(row['rank'])}. "
    return f"{rank}Long {row['long']} / Short {row['short']} ({row['peer_group']})"


# ---- charts' data --------------------------------------------------------

def spread_frame(spreads: pd.DataFrame, pair_id: str) -> pd.DataFrame:
    """Daily hedged spread for one pair, as a cumulative return in percent."""
    rows = spreads[spreads["pair_id"] == pair_id].sort_values("date")
    return pd.DataFrame({"date": pd.to_datetime(rows["date"]), "spread_pct": rows["spread"] * 100}).reset_index(drop=True)


def quarterly_ratios(quarterly: pd.DataFrame, cik10: str, ticker: str, count: int = 12) -> pd.DataFrame:
    """One row per (quarter, line) for the company's last `count` quarters:
    revenue growth vs the year-ago quarter, gross margin, operating margin
    and trailing-twelve-month free cash flow margin, as fractions."""
    rows = quarterly[quarterly["cik10"] == cik10]
    if rows.empty:
        return pd.DataFrame(columns=["ticker", "period_end", "quarter", "line", "value", "derived"])
    rows = rows.assign(pos=rows["fiscal_year"].astype("int64") * 4 + rows["quarter"].astype("int64"))
    val = rows.pivot_table(index="pos", columns="metric", values="val", aggfunc="last")
    ttm = rows.pivot_table(index="pos", columns="metric", values="val_ttm", aggfunc="last")
    revenue = rows[rows["metric"] == "revenue"].drop_duplicates("pos", keep="last").set_index("pos")

    def column(frame, name):
        return frame[name] if name in frame.columns else pd.Series(float("nan"), index=frame.index)

    rev = column(val, "revenue")
    year_ago = rev.reindex(rev.index - 4).set_axis(rev.index)
    ratios = pd.DataFrame({
        "revenue_growth": rev / year_ago.where(year_ago > 0) - 1,
        "gross_margin": column(val, "gross_profit") / rev.where(rev > 0),
        "operating_margin": column(val, "operating_income") / rev.where(rev > 0),
        "fcf_margin_ttm": (column(ttm, "operating_cash_flow") - column(ttm, "capex"))
        / column(ttm, "revenue").where(column(ttm, "revenue") > 0),
    }).loc[sorted(revenue.index)[-count:]]

    long = ratios.reset_index().melt(id_vars="pos", var_name="line", value_name="value").dropna(subset=["value"])
    meta = revenue[["fiscal_year", "quarter", "period_end", "method"]]
    long = long.join(meta, on="pos")
    return pd.DataFrame({
        "ticker": ticker,
        "period_end": pd.to_datetime(long["period_end"]),
        "quarter": [f"FY{int(y)} Q{int(q)}" for y, q in zip(long["fiscal_year"], long["quarter"])],
        "line": long["line"].map(QUARTER_LINES),
        "value": long["value"].astype(float),
        "derived": long["method"] != "reported",
    }).sort_values(["line", "period_end"]).reset_index(drop=True)


# ---- peer groups ---------------------------------------------------------

def group_view(scores: pd.DataFrame, metrics: pd.DataFrame, companies: pd.DataFrame | None,
               deals: pd.DataFrame | None, group: str) -> pd.DataFrame:
    """One peer group's companies, best composite first."""
    rows = scores[scores["peer_group"] == group].merge(
        metrics.drop(columns=["peer_group"], errors="ignore"), on="ticker", how="left", suffixes=("", "_m"))
    names = {} if companies is None else companies.set_index("ticker")["name"].to_dict()
    deal = {} if deals is None else deals.set_index("ticker")["status"].to_dict()
    rows = rows.sort_values(["rank_in_group", "ticker"], na_position="last")
    return pd.DataFrame({
        "Rank": rows["rank_in_group"],
        "Ticker": rows["ticker"],
        "Company": rows["ticker"].map(names),
        "Composite": rows["composite"],
        "Valuation": rows["valuation"],
        "Quality": rows["quality"],
        "Direction": rows["direction"],
        "EV/EBITDA": rows["ev_ebitda"],
        "EV/EBIT": rows["ev_ebit"],
        "P/E": rows["pe"],
        "FCF yield %": rows["fcf_yield"] * 100,
        "Gross margin %": rows["gross_margin"] * 100,
        "Operating margin %": rows["operating_margin"] * 100,
        "ROIC %": rows["roic"] * 100,
        "Revenue growth %": rows["revenue_growth"] * 100,
        "Deal": rows["ticker"].map(lambda t: DEAL_LABELS.get(deal.get(t, "none"), "")),
    }).reset_index(drop=True)


# ---- memos ---------------------------------------------------------------

def memo_body(memo: str) -> str:
    """The memo without its title line (the page shows the pair already)."""
    lines = memo.split("\n")
    return "\n".join(lines[1:]).lstrip("\n") if lines and lines[0].startswith("# ") else memo


def comparisons_view(fact_sheet_json: str) -> pd.DataFrame:
    facts = json.loads(fact_sheet_json)
    long, short = facts["long"], facts["short"]
    return pd.DataFrame([{
        "Pillar": c["pillar"].capitalize(), "Metric": c["label"], long: c["long"], short: c["short"],
        "Peer median": c.get("peer_median") or "n/a",
        "Stronger": long if c["edge"] > 0 else (short if c["edge"] < 0 else "level"),
    } for c in facts["comparisons"]])
