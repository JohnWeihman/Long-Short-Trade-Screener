from screener.data.rawfacts import (
    annual_facts,
    choose,
    filing_url,
    is_annual_duration,
    latest_fiscal_year_end,
    latest_year_summary,
    period_days,
)


def fact(val, end, start=None, form="10-K", filed="2025-02-20", accn="0000000001-25-000010", fy=2024, fp="FY"):
    row = {"end": end, "val": val, "accn": accn, "fy": fy, "fp": fp, "form": form, "filed": filed}
    if start:
        row["start"] = start
    return row


def company():
    """A company with a January fiscal year end, shaped like real companyfacts,
    including the things that make naive selection go wrong."""
    usd, shares = "USD", "shares"
    return {
        "cik": 1234,
        "entityName": "Test Semi Inc",
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {"units": {shares: [
                    fact(950, "2024-02-16", filed="2024-02-21", accn="A-24"),  # prior year's cover page
                    fact(1000, "2025-02-14"),
                    fact(1010, "2025-05-20", form="10-Q", filed="2025-05-25", accn="Q-25"),
                ]}},
            },
            "us-gaap": {
                # Older tag the company stopped using: must not win on priority alone.
                "Revenues": {"units": {usd: [
                    fact(300, "2023-01-29", "2022-01-31", filed="2023-02-24", accn="A-23", fy=2022),
                ]}},
                "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {usd: [
                    # Prior year, as first reported and as repeated in the next 10-K (fy=2024!).
                    fact(400, "2024-01-28", "2023-01-30", filed="2024-02-21", accn="A-24", fy=2023),
                    fact(400, "2024-01-28", "2023-01-30"),
                    # Latest year.
                    fact(500, "2025-01-26", "2024-01-29"),
                    # Fourth quarter and nine months that also end in an annual filing or share fy.
                    fact(140, "2025-01-26", "2024-10-28"),
                    fact(360, "2024-10-27", "2024-01-29", form="10-Q", filed="2024-11-20", accn="Q-24", fp="Q3"),
                    # A later 10-Q whose year-to-date must not be mistaken for a year.
                    fact(150, "2025-04-27", "2025-01-27", form="10-Q", filed="2025-05-25", accn="Q-25", fy=2025, fp="Q1"),
                ]}},
                "NetIncomeLoss": {"units": {usd: [
                    fact(90, "2025-01-26", "2024-01-29"),
                    # Restated in an amended filing.
                    fact(85, "2025-01-26", "2024-01-29", form="10-K/A", filed="2025-06-01", accn="AA-25"),
                ]}},
                "ProfitLoss": {"units": {usd: [fact(92, "2025-01-26", "2024-01-29")]}},
                "CommonStockSharesOutstanding": {"units": {shares: [
                    fact(940, "2024-01-28"),
                    fact(990, "2025-01-26"),
                ]}},
                "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {shares: [
                    fact(1005, "2025-01-26", "2024-01-29"),
                    fact(1002, "2025-01-26", "2024-10-28"),
                ]}},
            },
        },
    }


def test_period_days_counts_both_ends():
    assert period_days(fact(1, "2024-12-31", "2024-01-01")) == 366
    assert period_days(fact(1, "2024-12-31")) is None


def test_is_annual_duration():
    assert is_annual_duration(fact(1, "2025-01-26", "2024-01-29"))  # 52 weeks
    assert is_annual_duration(fact(1, "2024-02-03", "2023-01-29"))  # 53 weeks
    assert not is_annual_duration(fact(1, "2025-01-26", "2024-10-28"))  # a quarter inside a 10-K
    assert not is_annual_duration(fact(1, "2025-01-26"))  # point in time
    assert not is_annual_duration(fact(1, "2025-01-26", "2024-01-29", form="10-Q"))


def test_latest_fiscal_year_end_ignores_quarterly_filings():
    assert latest_fiscal_year_end(company()) == "2025-01-26"


def test_no_annual_data_gives_none():
    assert latest_fiscal_year_end({"facts": {}}) is None
    assert latest_year_summary({"facts": {}})["fiscal_year_end"] is None


def test_revenue_uses_the_tag_that_has_the_latest_year():
    revenue = latest_year_summary(company())["revenue"]
    assert revenue["chosen"]["val"] == 500
    assert revenue["chosen"]["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    # The 13-week figure with the same end date is not a candidate.
    assert [r["val"] for r in revenue["candidates"]] == [500]


def test_selection_is_by_period_not_by_fy_field():
    # The prior-year value was re-reported with fy=2024; asking for the prior
    # year end must return it, and asking for the latest must not.
    prior = annual_facts(company(), ("RevenueFromContractWithCustomerExcludingAssessedTax",), "2024-01-28")
    assert [r["val"] for r in prior] == [400, 400]
    assert [r["accn"] for r in prior] == ["A-24", "0000000001-25-000010"]


def test_restatement_picks_latest_filing_and_keeps_all_candidates():
    net_income = latest_year_summary(company())["net_income"]
    assert net_income["chosen"]["val"] == 85
    assert net_income["chosen"]["form"] == "10-K/A"
    assert [(r["tag"], r["val"]) for r in net_income["candidates"]] == [
        ("NetIncomeLoss", 90),
        ("NetIncomeLoss", 85),
        ("ProfitLoss", 92),
    ]


def test_share_counts():
    summary = latest_year_summary(company())
    # Cover page: the 10-K's own count, not last year's and not a later 10-Q's.
    assert summary["shares_cover_page"]["chosen"]["val"] == 1000
    assert summary["shares_cover_page"]["chosen"]["end"] == "2025-02-14"
    assert summary["shares_balance_sheet"]["chosen"]["val"] == 990
    assert summary["shares_diluted_weighted"]["chosen"]["val"] == 1005


def test_missing_metric_is_reported_as_absent():
    data = company()
    del data["facts"]["dei"]
    cover = latest_year_summary(data)["shares_cover_page"]
    assert cover == {"chosen": None, "candidates": []}


def test_choose_empty():
    assert choose([]) is None


def test_filing_url():
    assert filing_url("0001045810", "0001045810-25-000023") == (
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/0001045810-25-000023-index.htm"
    )
