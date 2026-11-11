"""Each company's own fiscal calendar, and where every reported period sits in it.

Fiscal years come from the full-year revenue and net income periods the
company reported in annual filings, never from EDGAR's fy/fp fields and
never from the calendar: NVDA's year ends in late January, MU's in late
August, and many use 52/53-week years.

A duration is then one of:
    Q    a single fiscal quarter (three months)
    YTD  year-to-date through quarter 2 or 3 (six or nine months)
    FY   the full fiscal year
and a point-in-time value (span 'I') belongs to the quarter whose end date
it carries. Anything else (stub periods, trailing twelve months, odd
ranges) is left unassigned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.data.rawfacts import ANNUAL_FORMS, MAX_ANNUAL_DAYS, MIN_ANNUAL_DAYS

ANCHOR_METRICS = ("revenue", "net_income")
START_TOLERANCE_DAYS = 7  # a year-to-date period starts on the fiscal year's first day
LENGTH_TOLERANCE_DAYS = 12  # a 14-week quarter is 7 days longer than a 13-week one
MIN_YEAR_GAP_DAYS = 300
COVER_WINDOW_DAYS = 150  # a cover-page count is dated between the period end and the filing
LABEL_SHIFT_DAYS = 15 # a year ending 3 January 2026 is fiscal 2025; one ending 25 January 2026 is fiscal 2026

FISCAL_YEAR_COLUMNS = ["cik10", "fiscal_year", "fy_start", "fy_end", "complete"]


def fiscal_years(metric_facts: pd.DataFrame) -> pd.DataFrame:
    """One row per company fiscal year, plus the year in progress.

    The year in progress starts the day after the last completed year, has
    no fy_end yet, and complete = False.
    """
    rows = metric_facts[metric_facts["metric"].isin(ANCHOR_METRICS) & metric_facts["start"].notna()]
    days = (rows["end"] - rows["start"]).dt.days + 1
    annual_form = rows["form_first"].isin(ANNUAL_FORMS) | rows["form_latest"].isin(ANNUAL_FORMS)
    rows = rows[days.between(MIN_ANNUAL_DAYS, MAX_ANNUAL_DAYS) & annual_form]

    out = []
    for cik, group in rows.groupby("cik10", sort=True):
        starts = group.groupby("end")["start"].agg(lambda s: s.mode().iloc[0]).sort_index()
        last_end = None
        for end, start in starts.items():
            if last_end is not None and (end - last_end).days < MIN_YEAR_GAP_DAYS:
                continue  # two "years" ending close together: keep the earlier
            out.append({"cik10": cik, "fy_start": start, "fy_end": end, "complete": True})
            last_end = end
        if last_end is not None:
            out.append({"cik10": cik, "fy_start": last_end + pd.Timedelta(days=1), "fy_end": pd.NaT, "complete": False})

    years = pd.DataFrame(out, columns=["cik10", "fy_start", "fy_end", "complete"])
    if years.empty:
        return pd.DataFrame(columns=FISCAL_YEAR_COLUMNS).astype(
            {"cik10": "str", "fy_start": "datetime64[ns]", "fy_end": "datetime64[ns]", "complete": bool})
    years = years.astype({"cik10": "str", "fy_start": "datetime64[ns]", "fy_end": "datetime64[ns]", "complete": bool})
    label = (years["fy_end"] - pd.Timedelta(days=LABEL_SHIFT_DAYS)).dt.year
    previous = label.groupby(years["cik10"]).shift(1)
    years["fiscal_year"] = label.where(years["complete"], previous + 1).astype("Int64")
    return years[FISCAL_YEAR_COLUMNS]


def _attach_year(frame: pd.DataFrame, years: pd.DataFrame) -> pd.DataFrame:
    """Join each row to the fiscal year its end date falls in (if any)."""
    ordered = frame.astype({"cik10": "str", "start": "datetime64[ns]", "end": "datetime64[ns]"})
    ordered = ordered.sort_values("end", kind="stable")
    joined = pd.merge_asof(
        ordered, years.sort_values("fy_start", kind="stable"),
        left_on="end", right_on="fy_start", by="cik10", direction="backward",
    )
    inside = joined["fy_start"].notna() & (joined["fy_end"].isna() | (joined["end"] <= joined["fy_end"]))
    return joined[inside]


def assign_durations(metric_facts: pd.DataFrame, years: pd.DataFrame) -> pd.DataFrame:
    """Durations that fit the fiscal calendar, with fiscal_year, quarter
    (the quarter the period ends in) and span ('Q', 'YTD' or 'FY')."""
    rows = _attach_year(metric_facts[metric_facts["start"].notna()], years)
    year_days = ((rows["fy_end"] - rows["fy_start"]).dt.days + 1).fillna(365.0)
    quarter_days = year_days / 4
    days = (rows["end"] - rows["start"]).dt.days + 1
    start_offset = (rows["start"] - rows["fy_start"]).dt.days
    end_offset = (rows["end"] - rows["fy_start"]).dt.days + 1

    quarter = np.rint(end_offset / quarter_days).clip(1, 4).astype(int)
    n_quarters = np.rint(days / quarter_days).astype(int)
    right_length = (days - n_quarters * quarter_days).abs() <= LENGTH_TOLERANCE_DAYS
    from_year_start = start_offset.abs() <= START_TOLERANCE_DAYS
    ends_on_quarter = (end_offset - quarter * quarter_days).abs() <= LENGTH_TOLERANCE_DAYS

    cumulative = right_length & from_year_start & ends_on_quarter & (n_quarters == quarter)
    single = (
        right_length & (n_quarters == 1)
        & ((start_offset - (quarter - 1) * quarter_days).abs() <= LENGTH_TOLERANCE_DAYS)
    )
    span = np.select(
        [cumulative & (quarter == 1), cumulative & (quarter == 4), cumulative, single],
        ["Q", "FY", "YTD", "Q"],
        default="",
    )
    rows = rows.assign(quarter=quarter, span=span)
    in_progress = ~rows["complete"].astype(bool)
    rows = rows[(rows["span"] != "") & ~((rows["span"] == "FY") & in_progress)]
    return rows.drop(columns=["fy_start", "fy_end", "complete"]).reset_index(drop=True)


def fiscal_quarters(durations: pd.DataFrame, years: pd.DataFrame) -> pd.DataFrame:
    """The end date of every fiscal quarter that has reported data:
    cik10, fiscal_year, quarter, period_end."""
    ends = (
        durations.groupby(["cik10", "fiscal_year", "quarter"])["end"]
        .agg(lambda s: s.mode().iloc[0]).rename("period_end").reset_index()
    )
    year_ends = years[years["complete"]].assign(quarter=4).rename(columns={"fy_end": "period_end"})
    ends = pd.concat([ends, year_ends[["cik10", "fiscal_year", "quarter", "period_end"]]], ignore_index=True)
    ends = ends.drop_duplicates(["cik10", "fiscal_year", "quarter"], keep="first")
    return ends.sort_values(["cik10", "fiscal_year", "quarter"], kind="stable").reset_index(drop=True)


def assign_instants(metric_facts: pd.DataFrame, quarters: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time values dated exactly on a fiscal quarter end (span 'I')."""
    rows = metric_facts[metric_facts["start"].isna()].astype({"end": "datetime64[ns]"})
    joined = rows.merge(quarters.rename(columns={"period_end": "end"}), on=["cik10", "end"], how="inner")
    return joined.assign(span="I")


def assign_covers(metric_facts: pd.DataFrame, quarters: pd.DataFrame) -> pd.DataFrame:
    """Cover-page values (span 'I'): dated shortly AFTER a period end, they
    belong to the most recent fiscal quarter ending before them."""
    rows = metric_facts[metric_facts["start"].isna()].astype({"cik10": "str", "end": "datetime64[ns]"})
    if rows.empty or quarters.empty:
        return rows.assign(fiscal_year=pd.NA, quarter=pd.NA, span="I").iloc[0:0]
    ends = quarters.astype({"cik10": "str", "period_end": "datetime64[ns]"})
    joined = pd.merge_asof(
        rows.sort_values("end", kind="stable"), ends.sort_values("period_end", kind="stable"),
        left_on="end", right_on="period_end", by="cik10", direction="backward",
        allow_exact_matches=False, tolerance=pd.Timedelta(days=COVER_WINDOW_DAYS),
    )
    joined = joined[joined["period_end"].notna()].drop(columns="period_end")
    return joined.assign(span="I")


def assign_periods(
    metric_facts: pd.DataFrame, cover_metrics: frozenset[str] | set[str] = frozenset()
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(assigned facts, fiscal years, fiscal quarters).

    cover_metrics names the metrics of kind 'cover' in the tag map.
    """
    is_cover = metric_facts["metric"].isin(cover_metrics)
    regular = metric_facts[~is_cover]
    years = fiscal_years(regular)
    durations = assign_durations(regular, years)
    quarters = fiscal_quarters(durations, years)
    parts = [durations, assign_instants(regular, quarters)]
    if is_cover.any():
        parts.append(assign_covers(metric_facts[is_cover], quarters))
    assigned = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else durations
    assigned = assigned.astype({"fiscal_year": "Int64", "quarter": "int64"})
    return assigned, years, quarters
