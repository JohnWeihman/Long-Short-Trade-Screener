import pytest

from screener.cleaning.audit import company_coverage, coverage_matrix, find_metric, fiscal_year_ends
from screener.cleaning.tagmap import Formula, Metric, TagMapError, Term, load_tag_map, parse_tag_map

from test_rawfacts import company, fact

FY = "2025-01-26"
PRIOR_FY = "2024-01-28"


def small_map(extra=None):
    raw = {
        "revenue": {"kind": "duration", "unit": "USD", "tags": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]},
        "net_income": {"kind": "duration", "unit": "USD", "tags": ["NetIncomeLoss", "ProfitLoss"]},
        "shares_outstanding": {"kind": "instant", "unit": "shares", "tags": ["CommonStockSharesOutstanding"]},
        "shares_outstanding_cover": {"kind": "cover", "unit": "shares", "tags": ["dei:EntityCommonStockSharesOutstanding"]},
        "buybacks": {"kind": "duration", "unit": "USD", "optional": True, "tags": ["PaymentsForRepurchaseOfCommonStock"]},
    }
    raw.update(extra or {})
    return parse_tag_map(raw)


def with_facts(extra):
    """The test company plus extra us-gaap facts: {tag: [fact, ...]} in USD."""
    data = company()
    for tag, rows in extra.items():
        data["facts"]["us-gaap"][tag] = {"units": {"USD": rows}}
    return data


def annual(val):
    return fact(val, FY, "2024-01-29")


def at_year_end(val):
    return fact(val, FY)


# ---- loading and validation ------------------------------------------

def test_the_real_tag_map_loads_and_has_the_core_metrics():
    tag_map = load_tag_map()
    assert {"revenue", "net_income", "total_assets", "operating_cash_flow", "shares_outstanding_cover"} <= set(tag_map)
    assert tag_map["revenue"].tags[0] == ("us-gaap", "Revenues")
    assert tag_map["shares_outstanding_cover"].tags == (("dei", "EntityCommonStockSharesOutstanding"),)
    assert tag_map["buybacks"].optional and not tag_map["revenue"].optional


def test_shares_issued_is_not_used_as_shares_outstanding():
    # Issued includes treasury shares (TXN: 1,741M issued vs 908M outstanding).
    assert ("us-gaap", "CommonStockSharesIssued") not in load_tag_map()["shares_outstanding"].tags


def test_parse_keeps_priority_order_and_defaults():
    metric = small_map()["revenue"]
    assert metric == Metric(
        "revenue", "duration", "USD",
        (("us-gaap", "Revenues"), ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax")),
        False,
        (),
    )


def test_parse_formula_terms_are_metrics_or_tags():
    tag_map = small_map({
        "gross_profit": {
            "kind": "duration", "unit": "USD", "tags": ["GrossProfit"],
            "formulas": [{"add": ["revenue"], "subtract": ["CostOfRevenue"], "add_if_present": ["dei:Something"]}],
        },
    })
    assert tag_map["gross_profit"].formulas == (
        Formula(
            add=(Term(True, "revenue"),),
            subtract=(Term(False, "CostOfRevenue", "us-gaap"),),
            add_if_present=(Term(False, "Something", "dei"),),
        ),
    )


@pytest.mark.parametrize("bad, message", [
    ({}, "empty"),
    ({"m": {"kind": "weekly", "unit": "USD", "tags": ["A"]}}, "kind"),
    ({"m": {"kind": "instant", "tags": ["A"]}}, "unit"),
    ({"m": {"kind": "instant", "unit": "USD", "tags": []}}, "at least one tag or formula"),
    ({"m": {"kind": "instant", "unit": "USD", "tags": ["A"], "sign": -1}}, "unknown keys"),
    ({"a": {"kind": "instant", "unit": "USD", "tags": ["X"]},
      "b": {"kind": "instant", "unit": "USD", "tags": ["X"]}}, "both a and b"),
    ({"m": {"kind": "instant", "unit": "USD", "formulas": [{"add": ["not_a_metric"]}]}}, "not a metric"),
    ({"m": {"kind": "instant", "unit": "USD", "formulas": [{"multiply": ["A"]}]}}, "unknown formula keys"),
    ({"m": {"kind": "instant", "unit": "USD", "formulas": [{"add": []}]}}, "no terms"),
    ({"a": {"kind": "instant", "unit": "USD", "formulas": [{"add": ["b"]}]},
      "b": {"kind": "instant", "unit": "USD", "formulas": [{"add": ["a"]}]}}, "loop"),
])
def test_parse_rejects_bad_maps(bad, message):
    with pytest.raises(TagMapError, match=message):
        parse_tag_map(bad)


# ---- tag selection ---------------------------------------------------

def test_fiscal_year_ends_newest_first_and_limited():
    assert fiscal_year_ends(company(), small_map(), 5) == [FY, PRIOR_FY, "2023-01-29"]
    assert fiscal_year_ends(company(), small_map(), 2) == [FY, PRIOR_FY]


def test_fallback_is_per_period_not_per_company():
    tag_map = small_map()
    # The company used Revenues in FY2022 and the ASC 606 tag afterwards.
    old = find_metric(company(), tag_map, "revenue", "2023-01-29")
    new = find_metric(company(), tag_map, "revenue", FY)
    assert (old["tag"], old["val"]) == ("Revenues", 300.0)
    assert (new["tag"], new["val"]) == ("RevenueFromContractWithCustomerExcludingAssessedTax", 500.0)


def test_first_priority_tag_wins_and_latest_filing_is_used():
    found = find_metric(company(), small_map(), "net_income", FY)
    assert (found["tag"], found["val"], found["accn"]) == ("NetIncomeLoss", 85.0, "AA-25")
    assert found["tags_with_data"] == "NetIncomeLoss, ProfitLoss"
    assert found["derived"] is False and found["formula"] is None


def test_instant_and_cover_kinds():
    tag_map = small_map()
    assert find_metric(company(), tag_map, "shares_outstanding", FY)["val"] == 990.0
    cover = find_metric(company(), tag_map, "shares_outstanding_cover", FY)
    assert (cover["val"], cover["taxonomy"]) == (1000.0, "dei")
    # Prior year's cover page belongs to the prior year only.
    assert find_metric(company(), tag_map, "shares_outstanding_cover", PRIOR_FY)["val"] == 950.0


def test_wrong_unit_is_not_a_match():
    data = company()
    data["facts"]["us-gaap"]["NetIncomeLoss"]["units"] = {
        "EUR": data["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"]
    }
    found = find_metric(data, small_map(), "net_income", FY)
    assert (found["tag"], found["val"]) == ("ProfitLoss", 92.0)


def test_missing_metric_is_reported_not_dropped():
    found = find_metric(company(), small_map(), "buybacks", FY)
    assert found["found"] is False and found["tag"] is None and found["optional"] is True
    assert found["val"] is None and found["derived"] is False


# ---- formulas --------------------------------------------------------

GROSS_PROFIT = {"gross_profit": {
    "kind": "duration", "unit": "USD", "tags": ["GrossProfit"],
    "formulas": [{"add": ["revenue"], "subtract": ["CostOfRevenue"]}],
}}


def test_formula_fills_a_metric_no_tag_carries():
    data = with_facts({"CostOfRevenue": [annual(200)]})
    found = find_metric(data, small_map(GROSS_PROFIT), "gross_profit", FY)
    assert (found["val"], found["derived"], found["formula"]) == (300.0, True, "revenue - CostOfRevenue")
    assert found["tag"] is None and found["accn"] is None


def test_a_reported_tag_always_beats_a_formula():
    data = with_facts({"CostOfRevenue": [annual(200)], "GrossProfit": [annual(310)]})
    found = find_metric(data, small_map(GROSS_PROFIT), "gross_profit", FY)
    assert (found["val"], found["derived"], found["tag"]) == (310.0, False, "GrossProfit")


def test_formula_needs_every_required_term():
    found = find_metric(company(), small_map(GROSS_PROFIT), "gross_profit", FY)  # no CostOfRevenue
    assert found["found"] is False and found["val"] is None


def test_formula_terms_use_the_right_period():
    # Cost of revenue exists only for the prior year: it must not leak into this one.
    data = with_facts({"CostOfRevenue": [fact(150, PRIOR_FY, "2023-01-30")]})
    tag_map = small_map(GROSS_PROFIT)
    assert find_metric(data, tag_map, "gross_profit", FY)["found"] is False
    assert find_metric(data, tag_map, "gross_profit", PRIOR_FY)["val"] == 250.0


def test_optional_terms_are_used_only_when_present():
    spec = {"da": {"kind": "duration", "unit": "USD", "formulas": [
        {"add": ["Depreciation"], "add_if_present": ["AmortizationOfIntangibleAssets"]},
    ]}}
    both = with_facts({"Depreciation": [annual(40)], "AmortizationOfIntangibleAssets": [annual(10)]})
    found = find_metric(both, small_map(spec), "da", FY)
    assert (found["val"], found["formula"]) == (50.0, "Depreciation + AmortizationOfIntangibleAssets")

    one = with_facts({"Depreciation": [annual(40)]})
    found = find_metric(one, small_map(spec), "da", FY)
    assert (found["val"], found["formula"]) == (40.0, "Depreciation")


def test_components_are_summed_not_first_match():
    # ADI-style: current part of long-term debt is zero, commercial paper is not.
    spec = {"short_term_debt": {"kind": "instant", "unit": "USD", "tags": ["DebtCurrent"], "formulas": [
        {"add_if_present": ["LongTermDebtCurrent", "ShortTermBorrowings", "CommercialPaper"]},
    ]}}
    data = with_facts({"LongTermDebtCurrent": [at_year_end(0)], "ShortTermBorrowings": [at_year_end(447)]})
    found = find_metric(data, small_map(spec), "short_term_debt", FY)
    assert (found["val"], found["formula"]) == (447.0, "LongTermDebtCurrent + ShortTermBorrowings")
    # With none of the pieces reported, the metric is missing, not zero.
    assert find_metric(company(), small_map(spec), "short_term_debt", FY)["found"] is False


def test_formulas_are_tried_in_order_and_can_subtract_optional_terms():
    spec = {"long_term_debt": {"kind": "instant", "unit": "USD", "tags": ["LongTermDebtNoncurrent"], "formulas": [
        {"add": ["NotReported"]},
        {"add": ["LongTermDebt"], "subtract_if_present": ["LongTermDebtCurrent"]},
    ]}}
    data = with_facts({"LongTermDebt": [at_year_end(26)], "LongTermDebtCurrent": [at_year_end(1)]})
    found = find_metric(data, small_map(spec), "long_term_debt", FY)
    assert (found["val"], found["formula"]) == (25.0, "LongTermDebt - LongTermDebtCurrent")


def test_formula_can_build_on_another_derived_metric():
    spec = dict(GROSS_PROFIT)
    spec["operating_income"] = {"kind": "duration", "unit": "USD", "formulas": [
        {"add": ["gross_profit"], "subtract": ["OperatingExpenses"]},
    ]}
    data = with_facts({"CostOfRevenue": [annual(200)], "OperatingExpenses": [annual(120)]})
    found = find_metric(data, small_map(spec), "operating_income", FY)
    assert (found["val"], found["formula"]) == (180.0, "gross_profit - OperatingExpenses")


# ---- coverage table --------------------------------------------------

def test_company_coverage_and_matrix():
    coverage = company_coverage(company(), small_map(), years=5)
    assert len(coverage) == 3 * 5  # 3 fiscal years x 5 metrics
    coverage.insert(0, "ticker", "TEST")
    matrix = coverage_matrix(coverage)
    assert matrix.loc["TEST"].to_dict() == {
        "revenue": 3, "net_income": 1, "shares_outstanding": 2, "shares_outstanding_cover": 2, "buybacks": 0,
    }
