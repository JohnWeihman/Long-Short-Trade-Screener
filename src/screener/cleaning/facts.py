"""Flatten the mapped tags out of raw companyfacts and de-duplicate them.

Two tables come out of this:

facts_raw   one row per reported fact: every filing's version of every period.
facts       one row per (company, tag, unit, period), carrying both the value
            as FIRST reported and the LATEST (restated) value, with the filing
            behind each. Later stages use the latest value by default; the
            first-reported value is what was known at the time.

Nothing is interpreted here: no fiscal periods, no tag fallback, no formulas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.cleaning.tagmap import Metric
from screener.data.rawfacts import tag_rows

RAW_COLUMNS = [
    "cik10", "taxonomy", "tag", "unit", "start", "end", "val",
    "accn", "form", "filed", "fy", "fp", "frame",
]
PERIOD_KEY = ["cik10", "taxonomy", "tag", "unit", "start", "end"]
FACT_COLUMNS = PERIOD_KEY + [
    "val_first", "accn_first", "form_first", "filed_first",
    "val_latest", "accn_latest", "form_latest", "filed_latest",
    "n_filings", "n_values", "restated", "scale_errors", "large_disagreement",
]
SERIES_KEY = ["cik10", "taxonomy", "tag", "unit"]

SCALE_FACTORS = (1e3, 1e6)  # units vs thousands, units vs millions
SCALE_TOLERANCE = 0.005  # 35,673,280 vs 35,673 is 1,000.008x: rounding, same number
LARGE_DISAGREEMENT = 10.0


# Forms that carry financial statements. Proxy statements (DEF 14A) are left
# out on purpose: their pay-versus-performance tables reuse us-gaap tags for
# rounded or differently defined figures (COHR's FY2026 proxy tags 787M as
# NetIncomeLoss; the 10-K says 805M, and 787M is the figure including
# minority interests). Filed later than the 10-K, they would otherwise win
# as the "latest" value.
STATEMENT_FORMS = frozenset({
    "10-K", "10-KT", "10-Q", "10-QT", "8-K", "20-F", "40-F", "6-K",
    "S-1", "S-4", "S-11", "F-1", "F-4", "10-12B", "10-12G",
})


def is_statement_form(form: object) -> bool:
    """True for periodic reports, current reports and registration
    statements, including their /A amendments."""
    return isinstance(form, str) and form.removesuffix("/A") in STATEMENT_FORMS


def mapped_tags(tag_map: dict[str, Metric]) -> list[tuple[str, str]]:
    """Every (taxonomy, tag) the tag map can read, including formula terms."""
    tags: dict[tuple[str, str], None] = {}
    for metric in tag_map.values():
        for key in metric.tags:
            tags[key] = None
        for formula in metric.formulas:
            for term in formula.terms():
                if not term.is_metric:
                    tags[(term.taxonomy, term.name)] = None
    return list(tags)


def flatten_company(companyfacts: dict, cik10: str, tags: list[tuple[str, str]]) -> pd.DataFrame:
    """All reported values of the given tags for one company, as filed."""
    rows = [
        {**row, "cik10": cik10}
        for taxonomy, tag in tags
        for row in tag_rows(companyfacts, taxonomy, tag)
    ]
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    for column in ("start", "end", "filed"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["val"] = pd.to_numeric(frame["val"], errors="coerce").astype("float64")
    frame["fy"] = pd.to_numeric(frame["fy"], errors="coerce").astype("Int64")
    return frame


def flag_scale_errors(raw: pd.DataFrame) -> pd.Series:
    """True for rows whose value is a wrong-scale copy of the same period.

    Some filings report a number in the wrong scale (thousands of shares
    tagged as shares), so two filings give the same period values exactly
    1,000x or 1,000,000x apart. That is an error in one of them, not a
    restatement, and the later filing is not always the right one. Of such a
    pair, the value further from the usual size of that tag for that company
    (median order of magnitude over all its periods) is flagged.

    Only corrects when another filing of the SAME period supplies the
    evidence. A lone outlier is left alone here; that is validation's job.
    """
    flags = pd.Series(False, index=raw.index)
    use = raw[raw["form"].map(is_statement_form) & raw["val"].notna() & (raw["val"] != 0)]
    if use.empty:
        return flags

    magnitude = np.log10(use["val"].abs())
    typical = magnitude.groupby([use[k] for k in SERIES_KEY]).transform("median")
    work = use[PERIOD_KEY + ["val"]].assign(off=(magnitude - typical).abs(), row=use.index)

    distinct = work.drop_duplicates(PERIOD_KEY + ["val"])
    n_values = distinct.groupby(PERIOD_KEY, dropna=False)["val"].transform("size")
    multi = distinct[n_values > 1]
    if multi.empty:
        return flags

    pairs = multi.merge(multi, on=PERIOD_KEY, suffixes=("", "_other"))
    ratio = pairs["val"] / pairs["val_other"]  # negative if the signs differ: never a match
    scaled = pd.Series(False, index=pairs.index)
    for factor in SCALE_FACTORS:
        for target in (factor, 1 / factor):
            scaled |= (ratio / target - 1).abs() <= SCALE_TOLERANCE
    wrong = pairs[scaled & (pairs["off"] > pairs["off_other"])]
    flags.loc[wrong["row"].unique()] = True

    # Every filing that repeats a flagged value for that period is flagged too.
    bad = work.loc[flags[work.index]][PERIOD_KEY + ["val"]].drop_duplicates().assign(bad=True)
    marked = work.merge(bad, on=PERIOD_KEY + ["val"], how="left")
    flags.loc[marked.loc[marked["bad"].eq(True), "row"]] = True
    return flags


def dedupe(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (company, tag, unit, period) with first-reported and latest values.

    'First' and 'latest' are by filing date, then accession number. A period
    is 'restated' when those two values differ. Only facts from forms that
    carry financial statements count (see STATEMENT_FORMS), and wrong-scale
    copies (see flag_scale_errors) are set aside and counted in scale_errors.

    large_disagreement marks periods whose remaining values differ by 10x or
    more: too big to be a normal restatement, so one filing is probably
    wrong, but there is no safe rule for which. Left for validation.
    """
    suspect = raw["scale_suspect"] if "scale_suspect" in raw.columns else flag_scale_errors(raw)
    raw = raw.assign(scale_suspect=suspect.astype(bool))
    raw = raw[raw["form"].map(is_statement_form)]
    if raw.empty:
        return pd.DataFrame(columns=FACT_COLUMNS)
    dropped = raw[raw["scale_suspect"]].groupby(PERIOD_KEY, dropna=False).size().rename("scale_errors")
    raw = raw[~raw["scale_suspect"]]

    ordered = raw.sort_values(PERIOD_KEY + ["filed", "accn"], kind="stable", na_position="first")
    ordered = ordered.assign(_abs=ordered["val"].abs().where(ordered["val"] != 0))
    grouped = ordered.groupby(PERIOD_KEY, dropna=False, sort=False)
    facts = grouped.agg(
        _max=("_abs", "max"),
        _min=("_abs", "min"),
        val_first=("val", "first"),
        accn_first=("accn", "first"),
        form_first=("form", "first"),
        filed_first=("filed", "first"),
        val_latest=("val", "last"),
        accn_latest=("accn", "last"),
        form_latest=("form", "last"),
        filed_latest=("filed", "last"),
        n_filings=("accn", "nunique"),
        n_values=("val", "nunique"),
    ).reset_index()
    facts["restated"] = facts["val_first"] != facts["val_latest"]
    facts["large_disagreement"] = (facts["_max"] / facts["_min"] >= LARGE_DISAGREEMENT).fillna(False)
    facts = facts.merge(dropped.reset_index(), on=PERIOD_KEY, how="left")
    facts["scale_errors"] = facts["scale_errors"].fillna(0).astype(int)
    return facts[FACT_COLUMNS]
