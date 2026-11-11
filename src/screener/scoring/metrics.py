"""Ratios computed from the fundamentals table: one row per company.

Ranked metrics (see config/scoring.yaml for which pillar uses which):

  valuation   ebitda_yield, ebit_yield, earnings_yield, fcf_yield, sales_yield
  quality     gross_margin, operating_margin, roic, net_debt_to_ebitda, net_debt_to_ebit
  direction   operating_margin_change, revenue_growth_acceleration, fcf_margin_change

Valuation is expressed as yields: a multiple such as EV/EBITDA is undefined
or misleading when earnings are negative, while EBITDA/EV is simply
negative and ranks last. The familiar multiples (ev_ebitda, ev_ebit, pe,
ev_sales) are kept for display only and are blank when not meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMNS = ["cik10", "ticker", "peer_group", "as_of", "fiscal_year", "quarter", "period_end"]
RANKED = [
    "ebitda_yield", "ebit_yield", "earnings_yield", "fcf_yield", "sales_yield",
    "gross_margin", "operating_margin", "roic", "net_debt_to_ebitda", "net_debt_to_ebit",
    "operating_margin_change", "revenue_growth_acceleration", "fcf_margin_change",
]
DISPLAY = ["ev_ebitda", "ev_ebit", "pe", "ev_sales", "revenue_growth", "revenue_growth_1y_ago"]
COLUMNS = ID_COLUMNS + RANKED + DISPLAY


def over_positive(top: pd.Series, bottom: pd.Series) -> pd.Series:
    """top / bottom, blank unless bottom is positive."""
    return (top / bottom).where(bottom > 0)


def leverage(net_debt: pd.Series, earnings: pd.Series) -> pd.Series:
    """Net debt as a multiple of earnings. Lower is better; negative means net cash.

    With no earnings to service it, any net debt is the worst case (infinite).
    A loss-making company with net cash has no meaningful figure (blank).
    """
    ratio = (net_debt / earnings).where(earnings > 0)
    no_earnings = earnings <= 0
    return ratio.mask(no_earnings & (net_debt > 0), np.inf)


def compute_metrics(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """One row per company, in the same order as the fundamentals table."""
    f = fundamentals.reindex(columns=list(dict.fromkeys(list(fundamentals.columns) + [
        "revenue_ttm", "revenue_ttm_1y_ago", "revenue_ttm_2y_ago", "gross_profit_ttm", "operating_income_ttm",
        "operating_income_ttm_1y_ago", "ebitda_ttm", "net_income_ttm", "fcf_ttm", "fcf_ttm_1y_ago",
        "market_cap", "enterprise_value", "net_debt", "roic", "fiscal_year", "quarter", "period_end",
    ])))
    number = lambda name: pd.to_numeric(f[name], errors="coerce")  # noqa: E731
    revenue, revenue_1y, revenue_2y = number("revenue_ttm"), number("revenue_ttm_1y_ago"), number("revenue_ttm_2y_ago")
    ebit, ebit_1y, ebitda = number("operating_income_ttm"), number("operating_income_ttm_1y_ago"), number("ebitda_ttm")
    net_income, fcf, fcf_1y = number("net_income_ttm"), number("fcf_ttm"), number("fcf_ttm_1y_ago")
    market_cap, ev, net_debt = number("market_cap"), number("enterprise_value"), number("net_debt")

    out = f[ID_COLUMNS].copy()
    # Valuation, as yields.
    out["ebitda_yield"] = over_positive(ebitda, ev)
    out["ebit_yield"] = over_positive(ebit, ev)
    out["earnings_yield"] = over_positive(net_income, market_cap)
    out["fcf_yield"] = over_positive(fcf, market_cap)
    out["sales_yield"] = over_positive(revenue, ev)
    # Quality.
    out["gross_margin"] = over_positive(number("gross_profit_ttm"), revenue)
    out["operating_margin"] = over_positive(ebit, revenue)
    out["roic"] = number("roic")
    out["net_debt_to_ebitda"] = leverage(net_debt, ebitda)
    out["net_debt_to_ebit"] = leverage(net_debt, ebit)
    # Direction: now versus the same twelve months a year earlier.
    growth = over_positive(revenue, revenue_1y) - 1
    growth_1y = over_positive(revenue_1y, revenue_2y) - 1
    out["operating_margin_change"] = out["operating_margin"] - over_positive(ebit_1y, revenue_1y)
    out["revenue_growth_acceleration"] = growth - growth_1y
    out["fcf_margin_change"] = over_positive(fcf, revenue) - over_positive(fcf_1y, revenue_1y)
    # Display only.
    out["ev_ebitda"] = over_positive(ev, ebitda).where(ev > 0)
    out["ev_ebit"] = over_positive(ev, ebit).where(ev > 0)
    out["pe"] = over_positive(market_cap, net_income)
    out["ev_sales"] = over_positive(ev, revenue).where(ev > 0)
    out["revenue_growth"] = growth
    out["revenue_growth_1y_ago"] = growth_1y
    return out[COLUMNS]
