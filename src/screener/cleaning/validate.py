"""Checks on the cleaned tables. Nothing is corrected or dropped here: each
finding is a row in the `validation` table saying what looks wrong and where,
so it can be traced back to the raw facts (scripts/trace_metric.py).

    python -m screener.cleaning.validate                  # summary + worst findings
    python -m screener.cleaning.validate --ticker COHR
    python -m screener.cleaning.validate --check quarters_vs_year --limit 50

Severity: error = the value is almost certainly wrong; warning = probably
inconsistent, look before relying on it; info = worth knowing.
"""

from __future__ import annotations

import argparse

import pandas as pd

from screener import config
from screener.cleaning.store import read_table
from screener.cleaning.tagmap import Metric
from screener.data.cache import Cache

COLUMNS = ["check", "severity", "cik10", "ticker", "metric", "fiscal_year", "quarter", "value", "expected", "detail"]
YEAR_KEY = ["cik10", "metric", "fiscal_year"]

SUM_TOLERANCE = 0.005  # four quarters vs the reported year
IDENTITY_TOLERANCE = 0.005  # gross profit = revenue - cost of revenue
BALANCE_TOLERANCE = 0.05  # assets vs liabilities + equity (minority interests sit in between)
JUMP_FACTOR = 5.0  # year-over-year change that is implausible for a size metric
RECAST_TOLERANCE = 0.10
EPS_TOLERANCE = 0.10
MIN_DOLLARS = 1e6  # ignore differences smaller than this

NON_NEGATIVE = frozenset({
    "revenue", "cost_of_revenue", "rd_expense", "sga_expense", "depreciation_amortization",
    "diluted_shares", "cash", "short_term_investments", "receivables", "inventory", "current_assets",
    "total_assets", "current_liabilities", "short_term_debt", "long_term_debt", "total_liabilities",
    "shares_outstanding", "shares_outstanding_cover", "capex", "stock_comp", "dividends_paid", "buybacks",
})
SIZE_METRICS = ("revenue", "cost_of_revenue", "total_assets", "diluted_shares", "depreciation_amortization")
SHARE_COUNT_METRICS = ("shares_outstanding", "shares_outstanding_cover")  # either one is enough


def _issues(rows: pd.DataFrame, check: str, severity: str) -> pd.DataFrame:
    out = rows.assign(check=check, severity=severity)
    for column in COLUMNS:
        if column not in out.columns:
            out[column] = None
    return out[COLUMNS]


def _wide(table: pd.DataFrame, metrics: list[str], key: list[str]) -> pd.DataFrame:
    rows = table[table["metric"].isin(metrics)]
    wide = rows.pivot_table(index=key, columns="metric", values="val", aggfunc="last")
    return wide.reindex(columns=metrics).reset_index()


def check_quarters_vs_year(quarterly: pd.DataFrame, annual: pd.DataFrame, tag_map: dict[str, Metric]) -> pd.DataFrame:
    """Four standalone quarters should add up to the reported year."""
    names = [n for n, m in tag_map.items() if m.kind == "duration" and m.combine == "sum"]
    rows = quarterly[quarterly["metric"].isin(names)]
    sums = rows.groupby(YEAR_KEY)["val"].agg(total="sum", n="size").reset_index()
    merged = sums[sums["n"] == 4].merge(annual[YEAR_KEY + ["val"]], on=YEAR_KEY)
    gap = (merged["total"] - merged["val"]).abs()
    bad = merged[(gap > SUM_TOLERANCE * merged["val"].abs()) & (gap > MIN_DOLLARS)]
    return _issues(
        bad.assign(value=bad["total"], expected=bad["val"],
                   detail="four quarters do not sum to the reported year"),
        "quarters_vs_year", "warning")


def check_coverage(
    quarterly: pd.DataFrame, annual: pd.DataFrame, years: pd.DataFrame, tag_map: dict[str, Metric]
) -> pd.DataFrame:
    """Required metrics missing for a completed year, or with fewer than four quarters."""
    required = [n for n, m in tag_map.items() if not m.optional]
    grid = years[years["complete"]][["cik10", "fiscal_year"]].merge(pd.DataFrame({"metric": required}), how="cross")
    have = annual[YEAR_KEY].assign(has_annual=True)
    counts = quarterly.groupby(YEAR_KEY).size().rename("n_quarters").reset_index()
    grid = grid.merge(have, on=YEAR_KEY, how="left").merge(counts, on=YEAR_KEY, how="left")
    grid["has_annual"] = grid["has_annual"] == True  # noqa: E712
    grid["n_quarters"] = grid["n_quarters"].fillna(0).astype(int)

    # A company needs one share count, not both.
    shares = grid[grid["metric"].isin(SHARE_COUNT_METRICS)]
    any_count = shares.groupby(["cik10", "fiscal_year"])["has_annual"].transform("any")
    grid = grid.drop(index=shares.index[any_count & ~shares["has_annual"]])

    missing = grid[~grid["has_annual"]]
    partial = grid[grid["has_annual"] & (grid["n_quarters"] < 4)]
    return pd.concat([
        _issues(missing.assign(detail="no value for the fiscal year"), "missing_annual", "warning"),
        _issues(partial.assign(value=partial["n_quarters"], expected=4,
                               detail="annual value exists but not all four quarters"),
                "missing_quarters", "info"),
    ], ignore_index=True)


def check_gross_profit(annual: pd.DataFrame) -> pd.DataFrame:
    """Reported gross profit should equal revenue minus cost of revenue."""
    reported = annual[(annual["metric"] == "gross_profit") & ~annual["formula_derived"]][["cik10", "fiscal_year"]]
    wide = _wide(annual, ["revenue", "cost_of_revenue", "gross_profit"], ["cik10", "fiscal_year"])
    wide = wide.merge(reported, on=["cik10", "fiscal_year"]).dropna()
    expected = wide["revenue"] - wide["cost_of_revenue"]
    gap = (wide["gross_profit"] - expected).abs()
    bad = wide[(gap > IDENTITY_TOLERANCE * wide["revenue"].abs()) & (gap > MIN_DOLLARS)]
    return _issues(
        bad.assign(metric="gross_profit", value=bad["gross_profit"], expected=expected[bad.index],
                   detail="gross profit differs from revenue - cost of revenue (tags may be on different bases)"),
        "gross_profit_identity", "warning")


def check_balance_sheet(annual: pd.DataFrame) -> pd.DataFrame:
    """Assets should be close to liabilities plus equity."""
    wide = _wide(annual, ["total_assets", "total_liabilities", "equity"], ["cik10", "fiscal_year"]).dropna()
    expected = wide["total_liabilities"] + wide["equity"]
    gap = (wide["total_assets"] - expected).abs()
    bad = wide[gap > BALANCE_TOLERANCE * wide["total_assets"].abs()]
    return _issues(
        bad.assign(metric="total_assets", value=bad["total_assets"], expected=expected[bad.index],
                   detail="assets differ from liabilities + equity by more than 5%"),
        "balance_sheet_identity", "warning")


def check_negative(quarterly: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """Values that cannot be negative."""
    out = []
    for table in (quarterly, annual):
        bad = table[table["metric"].isin(NON_NEGATIVE) & (table["val"] < 0)]
        out.append(_issues(bad.assign(value=bad["val"], detail="negative value for a metric that cannot be negative"),
                           "negative_value", "error"))
    return pd.concat(out, ignore_index=True)


def check_jumps(annual: pd.DataFrame) -> pd.DataFrame:
    """A size metric that changes 5x in a year is more likely a bad fact than a real change."""
    rows = annual[annual["metric"].isin(SIZE_METRICS)].sort_values(YEAR_KEY, kind="stable")
    previous = rows.groupby(["cik10", "metric"])["val"].shift(1)
    previous_year = rows.groupby(["cik10", "metric"])["fiscal_year"].shift(1)
    ratio = (rows["val"] / previous).abs()
    consecutive = (rows["fiscal_year"] - previous_year) == 1
    bad = rows[consecutive & ((ratio > JUMP_FACTOR) | (ratio < 1 / JUMP_FACTOR))]
    return _issues(
        bad.assign(value=bad["val"], expected=previous[bad.index],
                   detail="changed more than 5x from the prior year ('expected' is the prior year)"),
        "year_over_year_jump", "warning")


def check_recast(quarterly: pd.DataFrame, tag_map: dict[str, Metric]) -> pd.DataFrame:
    """A derived quarter whose latest and first-reported values disagree:
    the full year was re-presented later but the year-to-date figure was not."""
    names = [n for n, m in tag_map.items() if m.kind == "duration" and m.combine == "sum"]
    rows = quarterly[quarterly["metric"].isin(names) & (quarterly["method"] == "derived")].dropna(subset=["val", "val_first"])
    gap = (rows["val"] - rows["val_first"]).abs()
    bad = rows[(gap > RECAST_TOLERANCE * rows["val_first"].abs()) & (gap > MIN_DOLLARS)]
    return _issues(
        bad.assign(value=bad["val"], expected=bad["val_first"],
                   detail="derived quarter mixes restated and original figures ('expected' is the first-reported basis)"),
        "recast_in_derived_quarter", "warning")


def check_q4_eps(quarterly: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """Q4 EPS computed as income / shares vs annual EPS minus the first three quarters."""
    eps = quarterly[quarterly["metric"] == "eps_diluted"]
    q4 = eps[(eps["quarter"] == 4) & (eps["method"] == "ratio")]
    first_three = eps[eps["quarter"] < 4].groupby(["cik10", "fiscal_year"])["val"].agg(total="sum", n="size").reset_index()
    year = annual[annual["metric"] == "eps_diluted"][["cik10", "fiscal_year", "val"]].rename(columns={"val": "annual"})
    merged = q4.merge(first_three[first_three["n"] == 3], on=["cik10", "fiscal_year"]).merge(year, on=["cik10", "fiscal_year"])
    expected = merged["annual"] - merged["total"]
    gap = (merged["val"] - expected).abs()
    bad = merged[(gap > EPS_TOLERANCE * expected.abs()) & (gap > 0.05)]
    return _issues(
        bad.assign(value=bad["val"], expected=expected[bad.index],
                   detail="Q4 EPS (income / shares) differs from annual EPS minus Q1-Q3 "
                          "(preferred dividends, loss quarter, or share count change)"),
        "q4_eps", "warning")


def check_filing_conflicts(facts: pd.DataFrame, since: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """Periods where filings give values 10x or more apart, and wrong-scale
    values that were set aside."""
    cutoff = pd.to_datetime(facts["cik10"].map(since)).astype("datetime64[ns]")
    recent = facts[facts["end"].astype("datetime64[ns]") >= cutoff]  # no cutoff (NaT) -> not recent
    conflict = recent[recent["large_disagreement"]]
    fixed = recent[recent["scale_errors"] > 0]
    period = lambda f: f["start"].dt.strftime("%Y-%m-%d").fillna("") + " to " + f["end"].dt.strftime("%Y-%m-%d")  # noqa: E731
    return pd.concat([
        _issues(conflict.assign(metric=conflict["tag"], value=conflict["val_latest"], expected=conflict["val_first"],
                                detail="filings disagree by 10x or more for " + period(conflict)
                                       + " ('expected' is the first-reported value); latest value is in use"),
                "filings_disagree", "warning"),
        _issues(fixed.assign(metric=fixed["tag"], value=fixed["val_latest"],
                             detail="wrong-scale value in another filing was set aside for " + period(fixed)),
                "scale_error_set_aside", "info"),
    ], ignore_index=True)


def validate(tables: dict[str, pd.DataFrame], tag_map: dict[str, Metric], recent_years: int = 5) -> pd.DataFrame:
    """All findings for each company's last `recent_years` completed fiscal
    years (plus the year in progress)."""
    years = tables["fiscal_years"]
    complete = years[years["complete"]]
    last = complete.groupby("cik10")["fiscal_year"].max()
    first_kept = (last - recent_years + 1).rename("first_kept").reset_index()
    since = complete.merge(first_kept, on="cik10")
    since = since[since["fiscal_year"] >= since["first_kept"]].groupby("cik10")["fy_start"].min().to_dict()

    def recent(table: pd.DataFrame) -> pd.DataFrame:
        merged = table.merge(first_kept, on="cik10")
        return merged[merged["fiscal_year"] >= merged["first_kept"]].drop(columns="first_kept")

    quarterly, annual = recent(tables["metrics_quarterly"]), recent(tables["metrics_annual"])
    findings = pd.concat([
        check_negative(quarterly, annual),
        check_quarters_vs_year(quarterly, annual, tag_map),
        check_recast(quarterly, tag_map),
        check_q4_eps(quarterly, annual),
        check_gross_profit(annual),
        check_balance_sheet(annual),
        check_jumps(annual),
        check_filing_conflicts(tables["facts"], since),
        check_coverage(quarterly, annual, recent(years), tag_map),
    ], ignore_index=True)

    tickers = tables["companies"].set_index("cik10")["ticker"]
    findings["ticker"] = findings["cik10"].map(tickers)
    order = {"error": 0, "warning": 1, "info": 2}
    findings = findings.sort_values(
        ["severity", "check", "ticker", "metric", "fiscal_year", "quarter"],
        key=lambda s: s.map(order) if s.name == "severity" else s, kind="stable", na_position="last",
    )
    return findings[COLUMNS].reset_index(drop=True)


def summary(findings: pd.DataFrame) -> pd.DataFrame:
    """Findings per check: count and how many companies are affected."""
    if findings.empty:
        return pd.DataFrame(columns=["severity", "check", "findings", "companies"])
    out = findings.groupby(["severity", "check"], sort=False).agg(
        findings=("check", "size"), companies=("ticker", "nunique")).reset_index()
    order = {"error": 0, "warning": 1, "info": 2}
    return out.sort_values("severity", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)


def _format(row) -> str:
    def number(x):
        if x is None or pd.isna(x):
            return "-"
        return f"{x:,.2f}" if abs(x) < 1000 else f"{x:,.0f}"
    where = f"FY{int(row.fiscal_year)}" if pd.notna(row.fiscal_year) else ""
    if pd.notna(row.quarter):
        where += f" Q{int(row.quarter)}"
    return (f"  {str(row.ticker):<5} {str(row.metric)[:44]:<44} {where:<9} "
            f"value {number(row.value):>18}  expected {number(row.expected):>18}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ticker")
    parser.add_argument("--check")
    parser.add_argument("--limit", type=int, default=8, help="findings shown per check")
    args = parser.parse_args(argv)

    cache = Cache(config.data_dir())
    findings = read_table(cache, "validation")
    if args.ticker:
        findings = findings[findings["ticker"] == args.ticker.upper()]
    if args.check:
        findings = findings[findings["check"] == args.check]
    if findings.empty:
        print("No findings.")
        return 0

    print(summary(findings).to_string(index=False))
    for (severity, check), group in findings.groupby(["severity", "check"], sort=False):
        print(f"\n{check} [{severity}] - {group['detail'].iloc[0] if group['detail'].nunique() == 1 else 'see detail column'}")
        for row in group.head(args.limit).itertuples():
            print(_format(row))
        if len(group) > args.limit:
            print(f"  ... and {len(group) - args.limit} more (use --check {check} --limit N)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
