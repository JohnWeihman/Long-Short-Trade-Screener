"""Altair charts for the app.

Colours are the dataviz reference palette: categorical slot 1 (blue) for the
long, slot 2 (orange) for the short, in both light and dark steps. Colour
follows the role, never the ticker or its rank. Thin 2px lines, markers only
on hover, hairline solid grid, a crosshair tooltip on every chart, a legend
plus endpoint labels when there are two series, none for one.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

PALETTE = {
    "light": {"long": "#2a78d6", "short": "#eb6834", "ink": "#0b0b0b", "secondary": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"long": "#3987e5", "short": "#d95926", "ink": "#f0efec", "secondary": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _style(chart: alt.Chart, mode: str, height: int) -> alt.Chart:
    p = PALETTE[mode]
    return (
        chart.properties(height=height, background="transparent")
        .configure_axis(gridColor=p["grid"], gridDash=[], domainColor=p["axis"], tickColor=p["axis"],
                        labelColor=p["muted"], titleColor=p["secondary"], labelFont=FONT, titleFont=FONT,
                        labelFontSize=11, titleFontSize=11, titleFontWeight="normal")
        .configure_legend(labelColor=p["secondary"], titleColor=p["secondary"], labelFont=FONT,
                          titleFont=FONT, orient="top", title=None, symbolStrokeWidth=2)
        .configure_view(strokeWidth=0)
        .configure_text(font=FONT)
        .configure_title(anchor="start", color=p["secondary"], font=FONT, fontSize=12, fontWeight="normal", offset=6)
    )


def spread_chart(frame: pd.DataFrame, long: str, short: str, mode: str = "light", height: int = 280) -> alt.Chart:
    """The hedged spread over time: one series, so no legend."""
    p = PALETTE[mode]
    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    base = alt.Chart(frame).encode(x=alt.X("date:T", title=None, axis=alt.Axis(grid=False, format="%b %Y")))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=p["axis"], strokeWidth=1).encode(y="y:Q")
    line = base.mark_line(color=p["long"], strokeWidth=2).encode(
        y=alt.Y("spread_pct:Q", title=None, axis=alt.Axis(format="+.0f")))
    point = base.mark_point(color=p["long"], filled=True, size=72, stroke=p["ink"], strokeWidth=0).encode(
        y="spread_pct:Q", opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    rule = base.mark_rule(color=p["muted"], strokeWidth=1).encode(
        opacity=alt.condition(hover, alt.value(0.7), alt.value(0)),
        tooltip=[alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                 alt.Tooltip("spread_pct:Q", title="Cumulative spread %", format="+.1f")],
    ).add_params(hover)
    title = f"Long {long} minus hedged short {short}, cumulative %"
    return _style(alt.layer(zero, line, point, rule).properties(title=title), mode, height)


def trend_chart(frame: pd.DataFrame, line_label: str, long: str, short: str, mode: str = "light",
                height: int = 220) -> alt.Chart:
    """One quarterly line (e.g. gross margin) for both companies: two series,
    so a legend and a label at each line's last point."""
    p = PALETTE[mode]
    data = frame[frame["line"] == line_label].assign(role=lambda d: d["ticker"].map({long: "Long", short: "Short"}))
    data = data.assign(series=lambda d: d["role"] + " " + d["ticker"])
    domain = [f"Long {long}", f"Short {short}"]
    color = alt.Color("series:N", scale=alt.Scale(domain=domain, range=[p["long"], p["short"]]),
                      legend=alt.Legend(orient="top", title=None))
    hover = alt.selection_point(fields=["period_end"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    base = alt.Chart(data).encode(x=alt.X("period_end:T", title=None, axis=alt.Axis(grid=False, format="%b %Y")))
    y = alt.Y("value:Q", title=None, axis=alt.Axis(format=".0%"))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=p["axis"], strokeWidth=1).encode(y="y:Q")
    lines = base.mark_line(strokeWidth=2).encode(y=y, color=color)
    points = base.mark_point(filled=True, size=72).encode(
        y="value:Q", color=color, opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    last = data.sort_values("period_end").groupby("series", as_index=False).tail(1)
    labels = alt.Chart(last).mark_text(align="left", dx=6, fontSize=11, color=p["secondary"]).encode(
        x="period_end:T", y="value:Q", text="ticker:N")
    rule = base.mark_rule(color=p["muted"], strokeWidth=1).encode(
        opacity=alt.condition(hover, alt.value(0.7), alt.value(0)),
        tooltip=[alt.Tooltip("ticker:N", title="Company"), alt.Tooltip("quarter:N", title="Quarter"),
                 alt.Tooltip("period_end:T", title="Ended", format="%Y-%m-%d"),
                 alt.Tooltip("value:Q", title=line_label, format=".1%"),
                 alt.Tooltip("derived:N", title="Derived from year-to-date")],
    ).add_params(hover)
    return _style(alt.layer(zero, lines, points, labels, rule).properties(title=line_label), mode, height)
