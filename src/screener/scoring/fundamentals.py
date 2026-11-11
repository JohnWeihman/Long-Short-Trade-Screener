"""Per-company fundamentals as of a date: the ingredients every ratio is
built from (market cap, net debt, enterprise value, trailing-twelve-month
income and cash flow lines, EBITDA, free cash flow, ROIC).

Everything is "as of" a date: only quarterly figures whose filing date
(known_on) is on or before it are used, with the last price on or before
it. Run for today this gives current fundamentals; run for past dates it
gives history without look-ahead in WHICH periods are used.

basis='latest' uses each period's most recent (restated) value, which is
right for current scores. For a strict backtest use basis='first' (values
as first reported); 'latest' on a past date still sees later restatements.

Share counts and prices are both on today's share basis (Stage 2 restates
shares for splits; the price source adjusts prices), so market cap is right
for any as-of date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from screener.cleaning.tagmap import Metric

TTM_METRICS = (
    "revenue", "gross_profit", "operating_income", "depreciation_amortization",
    "net_income", "operating_cash_flow", "capex", "eps_diluted",
)
DEBT_AND_CASH = ("cash", "short_term_investments", "short_term_debt", "long_term_debt")
SHARE_SOURCES = ("shares_outstanding_cover", "shares_outstanding")  # preferred first


@dataclass(frozen=True)
class Settings:
    tax_rate: float = 0.21
    max_price_age_days: int = 7
    max_balance_sheet_lag_quarters: int = 2


def position(fiscal_year, quarter):
    """Quarters on one number line: consecutive quarters differ by 1."""
    return fiscal_year * 4 + quarter


def known_as_of(quarterly: pd.DataFrame, as_of: pd.Timestamp, basis: str = "latest") -> pd.DataFrame:
    """Quarterly rows that had been filed by as_of, with `value` on the chosen basis."""
    if basis not in ("latest", "first"):
        raise ValueError("basis must be 'latest' or 'first'")
    rows = quarterly[quarterly["known_on"].notna() & (quarterly["known_on"] <= as_of)]
    rows = rows.assign(
        value=rows["val"] if basis == "latest" else rows["val_first"],
        pos=position(rows["fiscal_year"].astype("int64"), rows["quarter"].astype("int64")),
    )
    return rows[rows["value"].notna()]


def trailing(series: pd.Series, pos: int, average: bool = False) -> float:
    """Sum (or mean) of the four quarters ending at pos; NaN unless all four exist."""
    window = series.reindex(range(pos - 3, pos + 1))
    if window.isna().any():
        return float("nan")
    return float(window.mean() if average else window.sum())


def latest_balance(series: pd.Series, pos: int, max_lag: int) -> tuple[float, int | None]:
    """(value at pos or the nearest earlier quarter within max_lag, quarters of lag)."""
    for lag in range(max_lag + 1):
        value = series.get(pos - lag)
        if value is not None and not pd.isna(value):
            return float(value), lag
    return float("nan"), None


def price_as_of(prices: pd.DataFrame | None, as_of: pd.Timestamp) -> tuple[float, pd.Timestamp | None]:
    """Last close on or before as_of."""
    if prices is None or prices.empty:
        return float("nan"), None
    rows = prices[(prices["date"] <= as_of) & prices["close"].notna()]
    if rows.empty:
        return float("nan"), None
    last = rows.iloc[-1]
    return float(last["close"]), pd.Timestamp(last["date"])


def company_fundamentals(
    rows: pd.DataFrame, tag_map: dict[str, Metric], prices: pd.DataFrame | None,
    as_of: pd.Timestamp, settings: Settings = Settings(),
) -> dict:
    """Fundamentals for one company from its as-of quarterly rows
    (the output of known_as_of, filtered to the company)."""
    by_metric = {
        name: group.drop_duplicates("pos", keep="last").set_index("pos")["value"]
        for name, group in rows.groupby("metric")
    }
    empty = pd.Series(dtype="float64")
    get = lambda name: by_metric.get(name, empty)  # noqa: E731
    notes: list[str] = []
    out: dict = {"as_of": as_of}

    revenue = get("revenue")
    if revenue.empty:
        return {**out, "notes": "no quarterly revenue"}
    pos = int(revenue.index.max())
    latest = rows[(rows["metric"] == "revenue") & (rows["pos"] == pos)].iloc[0]
    out.update(fiscal_year=int(latest["fiscal_year"]), quarter=int(latest["quarter"]),
               period_end=latest["period_end"], known_on=rows[rows["pos"] == pos]["known_on"].max())

    # Trailing twelve months, now and one and two years back.
    for name in TTM_METRICS:
        average = tag_map[name].combine == "average" if name in tag_map else False
        out[f"{name}_ttm"] = trailing(get(name), pos, average)
    for name in ("revenue", "operating_income", "operating_cash_flow", "capex"):
        out[f"{name}_ttm_1y_ago"] = trailing(get(name), pos - 4)
    out["revenue_ttm_2y_ago"] = trailing(get("revenue"), pos - 8)
    if np.isnan(out["revenue_ttm"]):
        notes.append("fewer than four consecutive quarters of revenue")

    out["ebitda_ttm"] = out["operating_income_ttm"] + out["depreciation_amortization_ttm"]
    if np.isnan(out["ebitda_ttm"]) and not np.isnan(out["operating_income_ttm"]):
        notes.append("no D&A: EBITDA unavailable")
    out["fcf_ttm"] = out["operating_cash_flow_ttm"] - out["capex_ttm"]
    out["fcf_ttm_1y_ago"] = out["operating_cash_flow_ttm_1y_ago"] - out["capex_ttm_1y_ago"]

    # Balance sheet at the latest quarter. Debt, cash and investments a company
    # does not report are taken as zero (and noted).
    lag_limit = settings.max_balance_sheet_lag_quarters
    for name in DEBT_AND_CASH + ("equity",):
        value, lag = latest_balance(get(name), pos, lag_limit)
        if lag is None and name != "equity":
            value = 0.0
            notes.append(f"{name} not reported: taken as 0")
        elif lag:
            notes.append(f"{name} is {lag} quarter(s) old")
        out[name] = value
    out["total_debt"] = out["short_term_debt"] + out["long_term_debt"]
    out["net_debt"] = out["total_debt"] - out["cash"] - out["short_term_investments"]

    # Shares: the most recent count of either kind; cover page wins a tie.
    best = None
    for rank, name in enumerate(SHARE_SOURCES):
        series = get(name)
        usable = series[series.index <= pos]
        if usable.empty:
            continue
        candidate = (int(usable.index.max()), -rank, name, float(usable.loc[usable.index.max()]))
        best = candidate if best is None or candidate[:2] > best[:2] else best
    out["shares"] = best[3] if best else float("nan")
    out["shares_source"] = best[2] if best else None
    if best and pos - best[0] > lag_limit:
        notes.append(f"share count is {pos - best[0]} quarters old")
    if not best:
        notes.append("no share count")

    price, price_date = price_as_of(prices, as_of)
    out.update(price=price, price_date=price_date)
    if price_date is None:
        notes.append("no price")
    elif (as_of - price_date).days > settings.max_price_age_days:
        notes.append(f"price is {(as_of - price_date).days} days old")

    out["market_cap"] = out["shares"] * price
    out["enterprise_value"] = out["market_cap"] + out["net_debt"]

    # ROIC: after-tax operating income over average invested capital.
    def invested_capital(at: int) -> float:
        parts = {name: latest_balance(get(name), at, lag_limit)[0] for name in DEBT_AND_CASH + ("equity",)}
        for name in DEBT_AND_CASH:
            if np.isnan(parts[name]):
                parts[name] = 0.0
        return (parts["equity"] + parts["short_term_debt"] + parts["long_term_debt"]
                - parts["cash"] - parts["short_term_investments"])

    out["invested_capital"] = invested_capital(pos)
    out["invested_capital_1y_ago"] = invested_capital(pos - 4)
    average_capital = np.nanmean([out["invested_capital"], out["invested_capital_1y_ago"]]) \
        if not np.isnan(out["invested_capital"]) else float("nan")
    out["nopat_ttm"] = out["operating_income_ttm"] * (1 - settings.tax_rate)
    if not np.isnan(average_capital) and average_capital <= 0:
        notes.append("invested capital is not positive: ROIC undefined")
        out["roic"] = float("nan")
    else:
        out["roic"] = out["nopat_ttm"] / average_capital

    out["notes"] = "; ".join(notes)
    return out


def build_fundamentals(
    quarterly: pd.DataFrame, companies: pd.DataFrame, peers: pd.DataFrame, tag_map: dict[str, Metric],
    prices: dict[str, pd.DataFrame], as_of: pd.Timestamp, settings: Settings = Settings(), basis: str = "latest",
) -> pd.DataFrame:
    """One row per company in the peer file. prices maps ticker -> price frame."""
    as_of = pd.Timestamp(as_of)
    known = known_as_of(quarterly, as_of, basis)
    ciks = companies.set_index("ticker")["cik10"].to_dict()
    records = []
    for peer in peers.itertuples():
        cik10 = ciks.get(peer.ticker)
        base = {"cik10": cik10, "ticker": peer.ticker, "peer_group": peer.peer_group, "basis": basis}
        if cik10 is None:
            records.append({**base, "as_of": as_of, "notes": "not in the cleaned company table"})
            continue
        rows = known[known["cik10"] == cik10]
        records.append({**base, **company_fundamentals(rows, tag_map, prices.get(peer.ticker), as_of, settings)})
    frame = pd.DataFrame(records)
    frame["peer_group"] = frame["peer_group"].astype(object).where(frame["peer_group"].notna(), None)
    return frame
