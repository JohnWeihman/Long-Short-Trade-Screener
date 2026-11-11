"""Apply the tag map to the de-duplicated facts: one value per
(company, metric, period), for every period the company ever reported.

For each period, the first tag in priority order that has a value wins; if
none does, the first formula that can be computed. Periods are whatever the
filings reported (quarters, year-to-date, full years, balance sheet dates);
making sense of them is the job of periods.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.cleaning.tagmap import Formula, Metric, Term

KEY = ["cik10", "start", "end"]
VALUES = ["val", "val_first", "filed_first", "filed_latest", "form_first", "form_latest"]
COLUMNS = ["cik10", "metric", "start", "end", "val", "val_first", "derived", "source",
           "filed_first", "filed_latest", "form_first", "form_latest"]


def _tag_frame(facts: pd.DataFrame, taxonomy: str, tag: str, metric: Metric) -> pd.DataFrame:
    """One tag's values, in the unit and period type the metric expects."""
    rows = facts[(facts["tag"] == tag) & (facts["taxonomy"] == taxonomy) & (facts["unit"] == metric.unit)]
    rows = rows[rows["start"].notna()] if metric.kind == "duration" else rows[rows["start"].isna()]
    return rows.rename(columns={"val_latest": "val"})[KEY + VALUES]


def _term_frame(facts: pd.DataFrame, tag_map: dict[str, Metric], metric: Metric, term: Term, done: dict) -> pd.DataFrame:
    if term.is_metric:
        return resolve_metric(facts, tag_map, term.name, done)[KEY + VALUES]
    return _tag_frame(facts, term.taxonomy, term.name, metric)


def _apply_formula(
    facts: pd.DataFrame, tag_map: dict[str, Metric], metric: Metric, formula: Formula, done: dict
) -> pd.DataFrame:
    """Rows the formula can compute: every required term present, and at
    least one term present. filed_* is the latest date among the terms used."""
    groups = (
        (formula.add, 1, True),
        (formula.subtract, -1, True),
        (formula.add_if_present, 1, False),
        (formula.subtract_if_present, -1, False),
    )
    merged: pd.DataFrame | None = None
    parts = []
    for terms, sign, required in groups:
        for term in terms:
            i = len(parts)
            frame = _term_frame(facts, tag_map, metric, term, done)[KEY + ["val", "val_first", "filed_first", "filed_latest"]]
            frame = frame.rename(columns={c: f"{c}_{i}" for c in frame.columns if c not in KEY})
            merged = frame if merged is None else merged.merge(frame, on=KEY, how="outer")
            parts.append((i, sign, required, term.label))
    assert merged is not None

    present = {i: merged[f"val_{i}"].notna() for i, *_ in parts}
    usable = pd.Series(False, index=merged.index)
    for flag in present.values():
        usable |= flag
    for i, _, required, _ in parts:
        if required:
            usable &= present[i]

    out = merged[KEY].copy()
    text = pd.Series("", index=merged.index)
    for column in ("val", "val_first"):
        out[column] = sum(sign * merged[f"{column}_{i}"].fillna(0.0) for i, sign, _, _ in parts)
    for i, sign, _, label in parts:
        text += np.where(present[i], (" + " if sign > 0 else " - ") + label, "")
    out["source"] = text.str.removeprefix(" + ").str.strip()
    out["derived"] = True
    for column in ("filed_first", "filed_latest"):
        out[column] = merged[[f"{column}_{i}" for i, *_ in parts]].max(axis=1)
    out["form_first"] = None
    out["form_latest"] = None
    return out[usable]


def resolve_metric(facts: pd.DataFrame, tag_map: dict[str, Metric], name: str, done: dict | None = None) -> pd.DataFrame:
    """Every period's value for one metric (columns: COLUMNS without 'metric')."""
    done = {} if done is None else done
    if name in done:
        return done[name]
    metric = tag_map[name]

    frames = []
    for priority, (taxonomy, tag) in enumerate(metric.tags):
        frame = _tag_frame(facts, taxonomy, tag, metric).assign(source=tag, derived=False, _priority=priority)
        frames.append(frame)
    if frames:
        chosen = pd.concat(frames, ignore_index=True).sort_values("_priority", kind="stable")
        chosen = chosen.drop_duplicates(KEY, keep="first").drop(columns="_priority")
    else:
        chosen = pd.DataFrame(columns=KEY + VALUES + ["source", "derived"])

    for formula in metric.formulas:
        computed = _apply_formula(facts, tag_map, metric, formula, done)
        if computed.empty:
            continue
        have = chosen[KEY].assign(_have=True)
        fresh = computed.merge(have, on=KEY, how="left")
        fresh = fresh[fresh["_have"].isna()].drop(columns="_have")
        chosen = pd.concat([chosen, fresh], ignore_index=True) if len(chosen) else fresh

    chosen = chosen.astype({"derived": bool}).reset_index(drop=True)
    done[name] = chosen
    return chosen


def resolve_all(facts: pd.DataFrame, tag_map: dict[str, Metric]) -> pd.DataFrame:
    """Every metric, every period: the long 'metric facts' table."""
    done: dict = {}
    frames = [resolve_metric(facts, tag_map, name, done).assign(metric=name) for name in tag_map]
    out = pd.concat(frames, ignore_index=True)
    for column in ("start", "end", "filed_first", "filed_latest"):
        out[column] = pd.to_datetime(out[column]).astype("datetime64[ns]")
    for column in ("val", "val_first"):
        out[column] = out[column].astype("float64")
    out["cik10"] = out["cik10"].astype("str")
    return out[COLUMNS].sort_values(["cik10", "metric", "end", "start"], kind="stable").reset_index(drop=True)
