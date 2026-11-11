"""Follow a cleaned number back to the raw EDGAR facts behind it.

For one metric value (a fiscal quarter or a fiscal year) this returns the
reported periods that went into it and, for each, every filing's version of
the underlying tags, so a wrong-looking number can be inspected before any
code is changed.
"""

from __future__ import annotations

import re

import pandas as pd

from screener.cleaning.periods import COVER_WINDOW_DAYS
from screener.cleaning.tagmap import Metric

RAW_VIEW = ["tag", "start", "end", "val", "val_reported", "split_factor", "scale_suspect", "form", "filed", "accn"]


def _same_period(frame: pd.DataFrame, start, end) -> pd.Series:
    same_start = frame["start"].isna() if pd.isna(start) else (frame["start"] == start)
    return same_start & (frame["end"] == end)


def leaf_tags(metric_facts: pd.DataFrame, tag_map: dict[str, Metric], cik10: str, metric: str, start, end) -> list[str]:
    """The raw tags behind one metric value: the tag itself, or the tags
    under every term of the formula that produced it."""
    rows = metric_facts[(metric_facts["cik10"] == cik10) & (metric_facts["metric"] == metric)]
    rows = rows[_same_period(rows, start, end)]
    if rows.empty:
        return []
    row = rows.iloc[0]
    if not row["derived"]:
        return [row["source"]]
    tags: list[str] = []
    for term in re.split(r" [+-] ", row["source"]):
        found = leaf_tags(metric_facts, tag_map, cik10, term, start, end) if term in tag_map else [term]
        tags.extend(t for t in found if t not in tags)
    return tags


def periods_behind(tables: dict[str, pd.DataFrame], tag_map: dict[str, Metric], cik10: str, metric: str,
                   fiscal_year: int, quarter: int | None) -> pd.DataFrame:
    """The reported periods a value was built from.

    quarter=None means the fiscal year. For a quarter this is every reported
    period of the metric ending on that quarter's end date and, when the
    quarter was derived, on the previous quarter's end date as well (the
    year-to-date figure that was subtracted).
    """
    facts = tables["metric_facts"]
    facts = facts[(facts["cik10"] == cik10) & (facts["metric"] == metric)]
    ends = tables["fiscal_quarters"]
    ends = ends[(ends["cik10"] == cik10) & (ends["fiscal_year"] == fiscal_year)].set_index("quarter")["period_end"]
    years = tables["fiscal_years"]
    year = years[(years["cik10"] == cik10) & (years["fiscal_year"] == fiscal_year)]
    if year.empty:
        return facts.iloc[0:0]
    fy_start = year["fy_start"].iloc[0]
    kind = tag_map[metric].kind

    wanted = [4] if quarter is None else [quarter]
    if quarter is not None and quarter > 1 and kind == "duration":
        table = tables["metrics_quarterly"]
        row = table[(table["cik10"] == cik10) & (table["metric"] == metric)
                    & (table["fiscal_year"] == fiscal_year) & (table["quarter"] == quarter)]
        if not row.empty and row["method"].iloc[0] != "reported":
            wanted.append(quarter - 1)
    dates = [ends[q] for q in wanted if q in ends.index]

    if kind == "cover":
        window = pd.Timedelta(days=COVER_WINDOW_DAYS)
        keep = pd.Series(False, index=facts.index)
        for date in dates:
            keep |= (facts["end"] > date) & (facts["end"] <= date + window)
        return facts[keep].sort_values("end")
    rows = facts[facts["end"].isin(dates)]
    if kind == "duration":
        rows = rows[rows["start"] >= fy_start - pd.Timedelta(days=7)]
        if quarter is None:
            rows = rows[(rows["start"] - fy_start).abs() <= pd.Timedelta(days=7)]
    return rows.sort_values(["end", "start"])


def raw_behind(tables: dict[str, pd.DataFrame], tag_map: dict[str, Metric], cik10: str, metric: str,
               periods: pd.DataFrame) -> pd.DataFrame:
    """Every filing's version of the tags behind the given periods."""
    raw = tables["facts_raw"]
    raw = raw[raw["cik10"] == cik10]
    out = []
    for period in periods.itertuples():
        tags = leaf_tags(tables["metric_facts"], tag_map, cik10, metric, period.start, period.end)
        rows = raw[raw["tag"].isin(tags) & _same_period(raw, period.start, period.end)]
        out.append(rows)
    if not out:
        return raw.iloc[0:0][RAW_VIEW]
    return pd.concat(out).sort_values(["end", "start", "tag", "filed", "accn"])[RAW_VIEW].reset_index(drop=True)
