"""Standalone fiscal quarters, fiscal years, and trailing twelve months.

Companies do not report the fourth quarter on its own, and cash flow
statements are year-to-date only. A quarter is therefore:

    reported   the company reported that three-month period directly
    derived    year-to-date through this quarter minus year-to-date through
               the previous one (Q4 = full year - nine months)
    ratio      numerator / denominator of two other metrics (Q4 EPS)

Reported always beats derived. Weighted averages (diluted shares) are
turned into quarter-weighted sums first, so Q4 = 4 x FY - (Q1 + Q2 + Q3),
which is an approximation.
"""

from __future__ import annotations

import pandas as pd

from screener.cleaning.tagmap import Metric

INDEX = ["cik10", "metric", "fiscal_year"]
QUARTERS = [1, 2, 3, 4]
META = ["source", "derived", "filed_first"]
QUARTERLY_COLUMNS = [
    "cik10", "metric", "fiscal_year", "quarter", "period_end", "val", "val_first",
    "method", "source", "formula_derived", "known_on", "val_ttm",
]
ANNUAL_COLUMNS = [
    "cik10", "metric", "fiscal_year", "period_end", "val", "val_first",
    "source", "formula_derived", "known_on",
]


def _wide(rows: pd.DataFrame, column: str, index: pd.MultiIndex) -> pd.DataFrame:
    """rows -> one column per quarter (1-4), one row per company/metric/year."""
    unique = rows.sort_values("end", kind="stable").drop_duplicates(INDEX + ["quarter"], keep="last")
    wide = unique.set_index(INDEX + ["quarter"])[column].unstack("quarter")
    return wide.reindex(index=index, columns=QUARTERS)


def _derive_additive(assigned: pd.DataFrame, tag_map: dict[str, Metric]) -> pd.DataFrame:
    """Quarters of the sum- and average-type duration metrics."""
    names = [n for n, m in tag_map.items() if m.kind == "duration" and m.combine in ("sum", "average")]
    rows = assigned[assigned["metric"].isin(names)]
    single = rows[rows["span"] == "Q"]
    cumulative = rows[rows["span"].isin(["YTD", "FY"])]
    index = pd.MultiIndex.from_frame(rows[INDEX].drop_duplicates()).sort_values()
    if len(index) == 0:
        return pd.DataFrame(columns=INDEX + ["quarter", "val", "val_first", "method"] + META)
    average = index.get_level_values("metric").map(lambda n: tag_map[n].combine == "average").to_numpy()

    meta_single = {m: _wide(single, m, index) for m in META}
    meta_cumulative = {m: _wide(cumulative, m, index) for m in META}

    # Only subtract like from like. The tag map falls back per period, so the
    # full year can come from one tag or formula and the nine months from
    # another (AMD's D&A); their difference is meaningless, so no quarter.
    source_q, source_y = meta_single["source"], meta_cumulative["source"]
    same_source = {}
    for k in (2, 3, 4):
        previous = source_q[1] if k == 2 else source_y[k - 1].where(source_y[k - 1].notna(), source_q[k - 1])
        same_source[k] = source_y[k] == previous

    frames = {}
    for column in ("val", "val_first"):
        q = _wide(single, column, index)
        y = _wide(cumulative, column, index)
        for k in (2, 3, 4):
            # An average over k quarters times k behaves like a year-to-date sum.
            y[k] = y[k].where(~average, y[k] * k)
        running = {1: q[1]}
        value = {1: q[1]}
        for k in (2, 3, 4):
            running[k] = y[k].where(y[k].notna(), running[k - 1] + q[k])
            difference = (running[k] - running[k - 1]).where(same_source[k])
            value[k] = q[k].where(q[k].notna(), difference)
        frames[column] = (q, pd.DataFrame(value))

    reported = frames["val"][0].notna()
    out = []
    for k in QUARTERS:
        part = pd.DataFrame({
            "val": frames["val"][1][k],
            "val_first": frames["val_first"][1][k],
            "method": reported[k].map({True: "reported", False: "derived"}),
        })
        for m in META:
            part[m] = meta_single[m][k].where(reported[k], meta_cumulative[m][k])
        out.append(part.assign(quarter=k).reset_index())
    result = pd.concat(out, ignore_index=True)
    return result[result["val"].notna()]


def _derive_ratio(assigned: pd.DataFrame, additive: pd.DataFrame, tag_map: dict[str, Metric]) -> pd.DataFrame:
    """Ratio-type metrics: the reported quarter, else numerator / denominator."""
    key = ["cik10", "fiscal_year", "quarter"]
    out = []
    for name, metric in tag_map.items():
        if metric.combine != "ratio" or metric.ratio_of is None:
            continue
        top, bottom = (additive[additive["metric"] == m][key + ["val", "val_first", "filed_first"]] for m in metric.ratio_of)
        grid = top.merge(bottom, on=key, suffixes=("_top", "_bottom"))
        rows = assigned[(assigned["metric"] == name) & (assigned["span"] == "Q")]
        rows = rows.sort_values("end", kind="stable").drop_duplicates(key, keep="last")
        rows = rows[key + ["val", "val_first"] + META]
        merged = grid.merge(rows, on=key, how="outer")
        reported = merged["val"].notna()
        for column in ("val", "val_first"):
            bottom_value = merged[f"{column}_bottom"].where(merged[f"{column}_bottom"] != 0)
            merged[column] = merged[column].where(reported, merged[f"{column}_top"] / bottom_value)
        merged["method"] = reported.map({True: "reported", False: "ratio"})
        merged["source"] = merged["source"].where(reported, f"{metric.ratio_of[0]} / {metric.ratio_of[1]}")
        merged["derived"] = merged["derived"].where(reported, False)
        merged["filed_first"] = merged["filed_first"].where(
            reported, merged[["filed_first_top", "filed_first_bottom"]].max(axis=1))
        merged["metric"] = name
        out.append(merged[INDEX + ["quarter", "val", "val_first", "method"] + META])
    if not out:
        return additive.iloc[0:0]
    result = pd.concat(out, ignore_index=True)
    return result[result["val"].notna()]


def _instants(assigned: pd.DataFrame) -> pd.DataFrame:
    rows = assigned[assigned["span"] == "I"]
    # Two cover pages can fall after the same quarter (a 10-K and its amendment): keep the later.
    rows = rows.sort_values("end", kind="stable").drop_duplicates(INDEX + ["quarter"], keep="last")
    return rows.assign(method="reported")[INDEX + ["quarter", "val", "val_first", "method"] + META]


def _trailing_twelve_months(quarterly: pd.DataFrame, tag_map: dict[str, Metric]) -> pd.Series:
    """Sum (or mean, for averages) of the four quarters ending at each row;
    missing unless all four are present and consecutive. Point-in-time
    metrics are their own value."""
    ordered = quarterly.sort_values(["cik10", "metric", "fiscal_year", "quarter"], kind="stable")
    position = ordered["fiscal_year"].astype("int64") * 4 + ordered["quarter"]
    groups = [ordered["cik10"], ordered["metric"]]
    total = ordered.groupby(groups)["val"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    consecutive = (position - position.groupby(groups).shift(3)) == 3
    total = total.where(consecutive)
    combine = ordered["metric"].map(lambda n: tag_map[n].combine)
    kind = ordered["metric"].map(lambda n: tag_map[n].kind)
    ttm = total.where(combine != "average", total / 4)
    return ttm.where(kind == "duration", ordered["val"]).reindex(quarterly.index)


def quarterly_table(assigned: pd.DataFrame, quarters: pd.DataFrame, tag_map: dict[str, Metric]) -> pd.DataFrame:
    """One row per (company, metric, fiscal quarter)."""
    additive = _derive_additive(assigned, tag_map)
    parts = [p for p in (additive, _derive_ratio(assigned, additive, tag_map), _instants(assigned)) if len(p)]
    if not parts:
        return pd.DataFrame(columns=QUARTERLY_COLUMNS)
    table = pd.concat(parts, ignore_index=True)
    table = table.merge(quarters, on=["cik10", "fiscal_year", "quarter"], how="left")
    table = table.rename(columns={"derived": "formula_derived", "filed_first": "known_on"})
    table["formula_derived"] = table["formula_derived"] == True  # noqa: E712  (missing -> False)
    table = table.sort_values(["cik10", "metric", "fiscal_year", "quarter"], kind="stable").reset_index(drop=True)
    table["val_ttm"] = _trailing_twelve_months(table, tag_map)
    return table[QUARTERLY_COLUMNS]


def annual_table(assigned: pd.DataFrame, quarters: pd.DataFrame) -> pd.DataFrame:
    """One row per (company, metric, completed fiscal year): the reported
    full-year figure, or the year-end balance for point-in-time metrics."""
    rows = assigned[(assigned["span"] == "FY") | ((assigned["span"] == "I") & (assigned["quarter"] == 4))]
    rows = rows.sort_values("end", kind="stable").drop_duplicates(INDEX, keep="last")
    year_ends = quarters[quarters["quarter"] == 4][["cik10", "fiscal_year", "period_end"]]
    rows = rows.merge(year_ends, on=["cik10", "fiscal_year"], how="left")
    rows = rows.rename(columns={"derived": "formula_derived", "filed_first": "known_on"})
    rows["formula_derived"] = rows["formula_derived"] == True  # noqa: E712  (missing -> False)
    return rows[ANNUAL_COLUMNS].sort_values(INDEX, kind="stable").reset_index(drop=True)
