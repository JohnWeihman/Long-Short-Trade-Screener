"""The Streamlit app: ranked pairs, pair detail, peer groups, data status.

    streamlit run streamlit_app.py

Reads only published/ (see screener/publish.py). Nothing here downloads data
or calls an API.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from screener.app import charts
from screener.app.data import (
    QUARTER_LINES,
    comparisons_view,
    group_view,
    load_manifest,
    load_table,
    memo_body,
    pair_label,
    pairs_view,
    published_dir,
    quarterly_ratios,
    spread_frame,
)

PAGES = ["Pairs", "Pair detail", "Peer groups", "Data status"]
TABLES = ["pairs", "pair_spreads", "deals", "scores", "metrics", "fundamentals", "companies",
          "metrics_quarterly", "memos", "validation", "model_runs", "pipeline_runs"]
REQUIRED = ["pairs", "scores", "metrics", "memos"]

NUMBER = st.column_config.NumberColumn
PAIR_COLUMNS = {
    "Rank": NUMBER(format="%d", width="small"),
    "Score gap": NUMBER(format="%.2f"), "Valuation gap": NUMBER(format="%+.2f"),
    "Quality gap": NUMBER(format="%+.2f"), "Direction gap": NUMBER(format="%+.2f"),
    "Correlation": NUMBER(format="%.2f"), "Hedge ratio": NUMBER(format="%.2f", help="$ short per $1 long"),
    "Spread z": NUMBER(format="%+.1f", help="Where the hedged spread is against its last year"),
    "Spread 3m %": NUMBER(format="%+.0f%%"), "Spread 12m %": NUMBER(format="%+.0f%%"),
}
GROUP_COLUMNS = {
    "Rank": NUMBER(format="%d", width="small"),
    **{c: NUMBER(format="%.2f") for c in ("Composite", "Valuation", "Quality", "Direction")},
    **{c: NUMBER(format="%.1fx") for c in ("EV/EBITDA", "EV/EBIT", "P/E")},
    **{c: NUMBER(format="%.1f%%") for c in ("FCF yield %", "Gross margin %", "Operating margin %", "ROIC %",
                                              "Revenue growth %")},
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_all(folder: str, stamp: str) -> dict:
    """Every published table (None where missing). `stamp` (the manifest's
    publish time) makes a new publish load fresh data."""
    return {name: load_table(folder, name) for name in TABLES}


def theme_mode() -> str:
    theme = getattr(st.context, "theme", None)
    return "dark" if getattr(theme, "type", None) == "dark" else "light"


def open_pair(pair_id: str) -> None:
    st.session_state["pair_id"] = pair_id
    st.session_state["page"] = "Pair detail"


# ---- pages ---------------------------------------------------------------

def page_pairs(t: dict) -> None:
    pairs = t["pairs"]
    st.header("Ranked pairs")
    st.caption("Long the higher-scoring company, short the lower, within a peer group. Scores run from 0 (worst in "
               "the group) to 1 (best); the gaps are the long's score minus the short's.")
    left, right = st.columns([3, 1])
    groups = left.multiselect("Peer groups", sorted(pairs["peer_group"].unique()), placeholder="All peer groups")
    include = right.toggle("Include pairs that fail a filter", value=False)
    view = pairs_view(pairs, include, groups)
    if view.empty:
        st.info("No pairs match.")
        return
    event = st.dataframe(view.drop(columns=["pair_id"]), column_config=PAIR_COLUMNS, hide_index=True,
                         height=min(36 * (len(view) + 1) + 4, 900),  # every row visible, no inner scroll
                         width="stretch", on_select="rerun", selection_mode="single-row", key="pairs_table")
    selected = event.selection.rows if event and hasattr(event, "selection") else []
    if selected:
        pair_id = view.iloc[selected[0]]["pair_id"]
        st.button(f"Open {pair_id.replace('/', ' / ')}", type="primary", on_click=open_pair, args=(pair_id,))
    else:
        st.caption("Select a row to open the pair.")


def page_pair(t: dict) -> None:
    pairs, memos = t["pairs"], t["memos"]
    candidates = pairs[pairs["passes"]].sort_values("rank")
    if candidates.empty:
        st.info("No pairs pass the filters today.")
        return
    ids = candidates["pair_id"].tolist()
    current = st.session_state.get("pair_id")
    index = ids.index(current) if current in ids else 0
    labels = {row["pair_id"]: pair_label(row) for _, row in candidates.iterrows()}
    pair_id = st.selectbox("Pair", ids, index=index, format_func=labels.get)
    st.session_state["pair_id"] = pair_id
    pair = candidates.set_index("pair_id").loc[pair_id]
    long, short = pair["long"], pair["short"]

    st.header(f"Long {long} / Short {short}")
    a, b, c, d = st.columns(4)
    a.metric("Score gap", f"{pair['gap']:.2f}", help="Long's composite score minus the short's (0-1 scale)")
    b.metric("Correlation", f"{pair['correlation']:.2f}", help="Weekly returns, last two years")
    c.metric("Hedge ratio", f"{pair['hedge_ratio']:.2f}", help=f"$ of {short} to short per $1 of {long}")
    d.metric("Spread, 12 months", "n/a" if pd.isna(pair["spread_chg_12m"]) else f"{pair['spread_chg_12m']:+.0%}",
             help="Long minus the hedged short; negative means the long has lagged")

    memo = memos.set_index("pair_id").loc[pair_id] if pair_id in set(memos["pair_id"]) else None
    has_ai = memo is not None and isinstance(memo.get("memo_ai"), str) and bool(memo["memo_ai"])
    tabs = st.tabs(["Memo", "AI memo", "Charts", "Against peers"])
    with tabs[0]:
        st.markdown(memo_body(memo["memo"]) if memo is not None else "No memo for this pair.")
    with tabs[1]:
        if has_ai:
            st.caption(f"Written by {memo['ai_model']} on {memo['ai_written']}; rewritten only when a company files.")
            st.markdown(memo_body(memo["memo_ai"]))
        else:
            reason = "" if memo is None else (memo.get("ai_note") or memo.get("ai_status") or "")
            st.info(f"No AI memo for this pair. {reason}".strip())
    with tabs[2]:
        chart_section(t, pair_id, long, short)
    with tabs[3]:
        if memo is not None:
            st.dataframe(comparisons_view(memo["fact_sheet"]), hide_index=True, width="stretch")
            st.caption("Peer median: the middle value across the peer group, both legs included.")


def chart_section(t: dict, pair_id: str, long: str, short: str) -> None:
    mode = theme_mode()
    spreads = t["pair_spreads"]
    if spreads is not None and pair_id in set(spreads["pair_id"]):
        frame = spread_frame(spreads, pair_id)
        st.subheader("Hedged spread")
        st.altair_chart(charts.spread_chart(frame, long, short, mode), width="stretch", theme=None)
        st.caption("Cumulative daily return of $1 long minus the hedge ratio's worth short, at today's hedge ratio. "
                   "It shows how the pair as it would be put on now has behaved; it is not a backtest.")
        with st.expander("Spread data"):
            st.dataframe(frame.rename(columns={"date": "Date", "spread_pct": "Cumulative spread %"}),
                         hide_index=True, width="stretch",
                         column_config={"Cumulative spread %": NUMBER(format="%+.2f")})
    quarterly, fundamentals = t["metrics_quarterly"], t["fundamentals"]
    if quarterly is None or fundamentals is None:
        return
    ciks = fundamentals.set_index("ticker")["cik10"].to_dict()
    frame = pd.concat([quarterly_ratios(quarterly, ciks.get(tk, ""), tk) for tk in (long, short)], ignore_index=True)
    if frame.empty:
        return
    st.subheader("Recent quarters")
    lines = [label for label in QUARTER_LINES.values() if label in set(frame["line"])]
    for left, right in zip(lines[0::2], lines[1::2] + [None]):
        cols = st.columns(2)
        for col, label in ((cols[0], left), (cols[1], right)):
            if label:
                col.altair_chart(charts.trend_chart(frame, label, long, short, mode), width="stretch",
                                 theme=None)
    st.caption("Quarters each company reported, on its own fiscal calendar. Some quarters are derived from "
               "year-to-date figures; hover a point to see which.")
    with st.expander("Quarterly data"):
        table = frame.pivot_table(index=["ticker", "quarter", "period_end"], columns="line", values="value").reset_index()
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={label: NUMBER(format="%.3f") for label in lines})


def page_groups(t: dict) -> None:
    scores, metrics = t["scores"], t["metrics"]
    st.header("Peer groups")
    groups = sorted(scores["peer_group"].dropna().unique())
    group = st.selectbox("Peer group", groups)
    view = group_view(scores, metrics, t["companies"], t["deals"], group)
    st.dataframe(view, column_config=GROUP_COLUMNS, hide_index=True, width="stretch")
    thin = scores[(scores["peer_group"] == group) & scores["thin_group"]]
    if len(thin):
        st.caption("This group is small, so ranks within it are coarse.")
    unplaced = scores[scores["peer_group"].isna()]["ticker"].tolist()
    if unplaced:
        st.caption(f"Not in any peer group, so not ranked: {', '.join(unplaced)}.")


def page_status(t: dict, manifest: dict | None) -> None:
    st.header("Data status")
    if manifest:
        a, b, c = st.columns(3)
        a.metric("Data as of", manifest.get("as_of") or "n/a")
        b.metric("Published", manifest.get("published_at", "n/a").replace(" UTC", ""), help="UTC")
        c.metric("Passing pairs", manifest.get("passing_pairs", "n/a"))
    runs = t["pipeline_runs"]
    st.subheader("Pipeline runs")
    if runs is None or runs.empty:
        st.caption("No pipeline runs recorded yet.")
    else:
        st.dataframe(runs.iloc[::-1], hide_index=True, width="stretch")
    ai = t["model_runs"]
    st.subheader("AI memo API calls")
    if ai is None or ai.empty:
        st.caption("No memo runs recorded yet.")
    else:
        st.caption(f"{int(ai['api_calls'].sum())} API calls in total, estimated ${ai['estimated_cost_usd'].sum():.4f}.")
        st.dataframe(ai.iloc[::-1], hide_index=True, width="stretch")
    deals = t["deals"]
    st.subheader("Merger and acquisition flags")
    if deals is not None:
        flagged = deals[deals["status"] != "none"]
        st.dataframe(flagged[["ticker", "status", "source", "filed", "evidence"]], hide_index=True,
                     width="stretch")
    findings = t["validation"]
    st.subheader("Data checks")
    if findings is not None and len(findings):
        counts = findings["severity"].value_counts().to_dict()
        st.caption(", ".join(f"{counts.get(s, 0)} {s}" for s in ("error", "warning", "info"))
                   + " over each company's last five fiscal years.")
        severity = st.multiselect("Severity", ["error", "warning", "info"], default=["error", "warning"])
        st.dataframe(findings[findings["severity"].isin(severity)], hide_index=True, width="stretch")


# ---- entry ---------------------------------------------------------------

def run() -> None:
    st.set_page_config(page_title="Pair screener", layout="wide")
    folder = published_dir()
    manifest = load_manifest(folder)
    tables = load_all(str(folder), (manifest or {}).get("published_at", ""))
    missing = [name for name in REQUIRED if tables.get(name) is None]
    if missing:
        st.title("Pair screener")
        st.error(f"No published data found in {folder} (missing: {', '.join(missing)}). "
                 "Run `python -m screener.pipeline --publish`, or `python -m screener.publish` after a pipeline run.")
        return

    st.sidebar.title("Pair screener")
    page = st.sidebar.radio("Page", PAGES, key="page", label_visibility="collapsed")
    if manifest:
        st.sidebar.caption(f"Data as of {manifest.get('as_of')}  \nPublished {manifest.get('published_at')}")
    st.sidebar.caption("A screening aid, not a recommendation.")

    if page == "Pairs":
        page_pairs(tables)
    elif page == "Pair detail":
        page_pair(tables)
    elif page == "Peer groups":
        page_groups(tables)
    else:
        page_status(tables, manifest)
