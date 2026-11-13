"""Write a memo for one pair from its fact sheet and the extracted filing text.

This is the template writer: a fixed structure filled with the fact sheet's
numbers and quoted excerpts from the filings. It adds no numbers and no
opinions of its own, so it is always available and always reproducible.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from screener.memos.sources import ANNUAL, EARNINGS

EXCERPT_CHARS = 450
DEAL_TEXT = {
    "pending_target": "is being acquired (pending deal)",
    "pending_other": "is party to a pending deal",
    "closed": "recently completed a deal, so its results may not be comparable with earlier periods",
}


def company_sources(ticker: str, sections: pd.DataFrame, documents: pd.DataFrame, risk_count: int = 3) -> dict:
    """The filing text a memo quotes for one company, with where it came from."""
    mine = sections[(sections["ticker"] == ticker) & sections["ok"]]
    docs = documents[documents["ticker"] == ticker]
    out = {"ticker": ticker, "release": None, "annual": None}

    releases = mine[mine["kind"] == EARNINGS]
    if len(releases):
        latest = releases[releases["filed"] == releases["filed"].max()].set_index("section")["text"]
        accn = releases.loc[releases["filed"].idxmax(), "accn"]
        doc = docs[docs["accn"] == accn].iloc[0]
        out["release"] = {
            "headline": latest.get("headline", ""), "outlook": latest.get("outlook", ""),
            "filed": f"{doc['filed']:%Y-%m-%d}", "accn": accn, "url": doc["url"],
        }
    annual = mine[mine["kind"] == ANNUAL]
    if len(annual):
        text = annual.set_index("section")["text"]
        accn = annual["accn"].iloc[0]
        doc = docs[docs["accn"] == accn].iloc[0]
        headings = [h for h in text.get("risk_headings", "").split("\n") if h][:risk_count]
        out["annual"] = {"risk_headings": headings, "filed": f"{doc['filed']:%Y-%m-%d}", "accn": accn, "url": doc["url"]}
    return out


def _excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """The start of a passage, cut at a sentence or word boundary."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit]
    stop = max(cut.rfind(". "), cut.rfind("; "))
    cut = cut[:stop + 1] if stop > limit * 0.5 else cut[:cut.rfind(" ")]
    return cut.rstrip(" ;,") + " ..."


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:+.0%}"


def _lead(value_long, value_short, long: str, short: str) -> str:
    if value_long is None or value_short is None:
        return "not scored for both"
    if value_long == value_short:
        return "level"
    return f"{long if value_long > value_short else short} ahead"


def render_template(facts: dict, long_sources: dict, short_sources: dict) -> str:
    """The memo as Markdown."""
    long, short = facts["long"], facts["short"]
    L, S = facts["companies"]["long"], facts["companies"]["short"]
    trade = facts["trade"]
    rank = f"Rank {facts['rank']} of {facts['passing_pairs']} passing pairs" if facts["rank"] else "Does not pass the filters"
    lines = [
        f"# Long {long} / Short {short}",
        f"{facts['peer_group']} | as of {facts['as_of']} | {rank}",
        "",
        f"**Composite score {L['composite']:.2f} vs {S['composite']:.2f} (gap {facts['composite_gap']:.2f}).** "
        f"{long} ranks {L['rank_in_group']} of {L['group_size']} in its peer group; {short} ranks {S['rank_in_group']}. "
        "Scores run from 0 (worst in the group) to 1 (best).",
        "",
        "## Why this pair",
        "",
        "| Pillar | " + long + " | " + short + " | |",
        "|---|---|---|---|",
    ]
    for pillar in ("valuation", "quality", "direction"):
        a, b = L["pillars"][pillar], S["pillars"][pillar]
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
        lines.append(f"| {pillar.capitalize()} | {fmt(a)} | {fmt(b)} | {_lead(a, b, long, short)} |")

    lines += ["", f"What favors {long} over {short}:", ""]
    lines += [f"- {c['label']}: {c['long']} vs {c['short']}" for c in facts["for_the_trade"]] or ["- nothing stands out"]
    if facts["against_the_trade"]:
        lines += ["", f"What favors {short}:", ""]
        lines += [f"- {c['label']}: {c['long']} vs {c['short']}" for c in facts["against_the_trade"]]

    lines += ["", "## What each company said", ""]
    for company, sources in ((L, long_sources), (S, short_sources)):
        release = sources["release"]
        if release is None:
            lines += [f"**{company['ticker']}**: no earnings release could be read.", ""]
            continue
        lines.append(f"**{company['ticker']}** ({release['filed']}): {release['headline']}")
        if release["outlook"]:
            lines += ["", f"> {_excerpt(release['outlook'])}"]
        else:
            lines += ["", "No outlook or guidance passage was found in the release."]
        lines.append("")

    lines += ["## Risks and caveats", ""]
    for company in (L, S):
        if company["deal_status"] in DEAL_TEXT:
            lines.append(f"- **{company['ticker']} {DEAL_TEXT[company['deal_status']]}.** "
                         f"{_excerpt(company['deal_evidence'], 260)}")
    if facts["thin_group"]:
        lines.append("- The peer group is small, so ranks within it are coarse.")
    z, year = trade["spread_z"], trade["spread_chg_12m"]
    if year is not None:
        moved = "outperformed" if year > 0 else "underperformed"
        lines.append(f"- Over the past 12 months {long} has {moved} the hedged {short} position by {abs(year):.0%} "
                     f"(3 months: {_pct(trade['spread_chg_3m'])}); the spread's z-score against the last year is "
                     f"{'n/a' if z is None else f'{z:+.1f}'}.")
    for company in (L, S):
        if company["data_notes"]:
            lines.append(f"- Data note, {company['ticker']}: {company['data_notes']}.")
    for company, sources in ((L, long_sources), (S, short_sources)):
        annual = sources["annual"]
        if annual and annual["risk_headings"]:
            lines += ["", f"Risks {company['ticker']} lists first in its 10-K:", ""]
            lines += [f"- {heading}" for heading in annual["risk_headings"]]

    lines += [
        "", "## Trade construction", "",
        f"- Weekly return correlation {trade['correlation']:.2f} over {trade['weeks']} weeks.",
        f"- Betas to the universe index: {long} {trade['beta_long']:.2f}, {short} {trade['beta_short']:.2f}.",
        f"- Hedge ratio {trade['hedge_ratio']:.2f}: short ${trade['hedge_ratio']:.2f} of {short} per $1.00 of {long}, "
        f"or {trade['share_ratio']:.2f} {short} shares per {long} share at ${L['price']:,.2f} and ${S['price']:,.2f}.",
        f"- Market cap: {long} ${L['market_cap_bn']:,.1f}B, {short} ${S['market_cap_bn']:,.1f}B. "
        f"Latest quarter: {long} {L['latest_quarter']} (ended {L['quarter_ended']}), "
        f"{short} {S['latest_quarter']} (ended {S['quarter_ended']}).",
        "", "## Sources", "",
    ]
    for sources in (long_sources, short_sources):
        if sources["release"]:
            r = sources["release"]
            lines.append(f"- {sources['ticker']} earnings release, filed {r['filed']}: [{r['accn']}]({r['url']})")
        if sources["annual"]:
            a = sources["annual"]
            lines.append(f"- {sources['ticker']} 10-K, filed {a['filed']}: [{a['accn']}]({a['url']})")
    lines += [
        f"- Scores, ratios and pair statistics: screener tables as of {facts['as_of']}.",
        "", "*Generated from SEC filings and screener data by a fixed template. A screening aid, not a recommendation.*",
    ]
    return "\n".join(lines)


def inputs_hash(facts: dict, long_sources: dict, short_sources: dict) -> str:
    """Changes whenever anything the memo is built from changes."""
    payload = json.dumps([facts, long_sources, short_sources], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
