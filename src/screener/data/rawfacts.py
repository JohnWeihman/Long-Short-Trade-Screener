"""Read-only lookups on a raw companyfacts JSON, for checking numbers by hand.

This is deliberately not the Stage 2 cleaning layer: it uses a short explicit
tag list and returns every candidate it finds instead of silently picking one.

Facts are selected by their period (start/end dates), never by the fy/fp
fields, which describe the filing a fact appeared in rather than the fact.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

ANNUAL_FORMS = ("10-K", "10-K/A", "10-KT")

# In priority order.
REVENUE_TAGS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
)
NET_INCOME_TAGS = (
    "NetIncomeLoss",
    "ProfitLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
)
DILUTED_SHARES_TAGS = ("WeightedAverageNumberOfDilutedSharesOutstanding",)
BALANCE_SHEET_SHARES_TAGS = ("CommonStockSharesOutstanding",)
COVER_SHARES_TAG = "EntityCommonStockSharesOutstanding"  # dei taxonomy

# A fiscal year is 52 or 53 weeks, or a calendar year.
MIN_ANNUAL_DAYS, MAX_ANNUAL_DAYS = 340, 380
# The cover-page share count is dated between year end and the filing date.
COVER_WINDOW_DAYS = 150

Row = dict[str, Any]


def tag_rows(companyfacts: dict, taxonomy: str, tag: str) -> list[Row]:
    """Every reported value of one tag, flattened across units."""
    concept = companyfacts.get("facts", {}).get(taxonomy, {}).get(tag)
    if not concept:
        return []
    return [
        {**row, "taxonomy": taxonomy, "tag": tag, "unit": unit}
        for unit, rows in concept.get("units", {}).items()
        for row in rows
    ]


def period_days(row: Row) -> int | None:
    if not row.get("start") or not row.get("end"):
        return None
    return (date.fromisoformat(row["end"]) - date.fromisoformat(row["start"])).days + 1


def is_annual_duration(row: Row) -> bool:
    """A full-fiscal-year duration reported in an annual filing."""
    days = period_days(row)
    return (
        row.get("form") in ANNUAL_FORMS
        and days is not None
        and MIN_ANNUAL_DAYS <= days <= MAX_ANNUAL_DAYS
    )


def _filed_order(row: Row) -> tuple[str, str]:
    return (row.get("filed", ""), row.get("accn", ""))


def annual_facts(companyfacts: dict, tags: tuple[str, ...], fy_end: str) -> list[Row]:
    """Full-year values ending on fy_end, in tag priority order, then by filing date."""
    found: list[Row] = []
    for tag in tags:
        rows = [
            r for r in tag_rows(companyfacts, "us-gaap", tag)
            if is_annual_duration(r) and r["end"] == fy_end
        ]
        found.extend(sorted(rows, key=_filed_order))
    return found


def instant_facts(
    companyfacts: dict, taxonomy: str, tags: tuple[str, ...], earliest: str, latest: str
) -> list[Row]:
    """Point-in-time values from annual filings dated within [earliest, latest]."""
    found: list[Row] = []
    for tag in tags:
        rows = [
            r for r in tag_rows(companyfacts, taxonomy, tag)
            if not r.get("start")
            and r.get("form") in ANNUAL_FORMS
            and earliest <= r.get("end", "") <= latest
        ]
        found.extend(sorted(rows, key=_filed_order))
    return found


def latest_fiscal_year_end(companyfacts: dict) -> str | None:
    """End date of the most recent full fiscal year reported in an annual filing."""
    ends = [
        row["end"]
        for tag in REVENUE_TAGS + NET_INCOME_TAGS
        for row in tag_rows(companyfacts, "us-gaap", tag)
        if is_annual_duration(row)
    ]
    return max(ends) if ends else None


def choose(rows: list[Row]) -> Row | None:
    """The highest-priority tag's most recently filed value.

    rows must come from annual_facts / instant_facts (priority, then filing order).
    """
    if not rows:
        return None
    first_tag = rows[0]["tag"]
    return [r for r in rows if r["tag"] == first_tag][-1]


def _metric(rows: list[Row]) -> dict[str, Any]:
    return {"chosen": choose(rows), "candidates": rows}


def latest_year_summary(companyfacts: dict) -> dict[str, Any]:
    """Revenue, net income and share counts for the latest fiscal year, with
    every candidate fact kept alongside the chosen one."""
    fy_end = latest_fiscal_year_end(companyfacts)
    summary: dict[str, Any] = {
        "entity": companyfacts.get("entityName"),
        "cik": companyfacts.get("cik"),
        "fiscal_year_end": fy_end,
    }
    if fy_end is None:
        return summary

    day_after = (date.fromisoformat(fy_end) + timedelta(days=1)).isoformat()
    window_end = (date.fromisoformat(fy_end) + timedelta(days=COVER_WINDOW_DAYS)).isoformat()
    summary.update(
        revenue=_metric(annual_facts(companyfacts, REVENUE_TAGS, fy_end)),
        net_income=_metric(annual_facts(companyfacts, NET_INCOME_TAGS, fy_end)),
        shares_cover_page=_metric(
            instant_facts(companyfacts, "dei", (COVER_SHARES_TAG,), day_after, window_end)
        ),
        shares_balance_sheet=_metric(
            instant_facts(companyfacts, "us-gaap", BALANCE_SHEET_SHARES_TAGS, fy_end, fy_end)
        ),
        shares_diluted_weighted=_metric(annual_facts(companyfacts, DILUTED_SHARES_TAGS, fy_end)),
    )
    return summary


def filing_url(cik: int | str, accession: str) -> str:
    """Link to the filing's index page on sec.gov."""
    return (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession.replace('-', '')}/{accession}-index.htm"
    )
