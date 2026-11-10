"""Interactive Plotly figures for the website. Same conventions as plots.py:
fixed A/B colours + distinct markers, delta = B - A (positive: A faster), titles say WHAT, not what it means."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COL_A, COL_B = "#0072B2", "#D55E00"
FILL_A, FILL_B = "rgba(0,114,178,0.35)", "rgba(213,94,0,0.35)"
GREY = "#9a9a9a"
COMPOUND_COL = {"SOFT": "#D62728", "MEDIUM": "#E6B800", "HARD": "#6b6b6b",
                "INTERMEDIATE": "#2CA02C", "WET": "#1F77B4", "UNKNOWN": "#999999"}


def _layout(fig, title, height=420, **kw):
    """Title on top; legend BELOW the plot at a fixed pixel offset so it never collides with titles/axis labels."""
    top, bottom = 70, 130
    plot_h = max(height - top - bottom, 100)
    fig.update_layout(title=dict(text=title, font=dict(size=14), x=0, xanchor="left", y=0.97, yanchor="top"),
                      height=height, margin=dict(l=60, r=20, t=top, b=bottom),
                      legend=dict(orientation="h", x=0, xanchor="left", y=-80 / plot_h, yanchor="top"), **kw)
    return fig


def _hover(df: pd.DataFrame, reason: bool = False) -> list[str]:
    out = []
    for r in df.itertuples():
        s = f"<b>{r.Driver}</b> lap {int(r.LapNumber)}<br>{r.LapTimeS:.3f} s<br>{r.Compound}, tyre age {r.TyreAge:.0f}"
        if bool(getattr(r, "InTraffic", False)):
            s += "<br>in traffic (proxy)"
        if reason and r.ExclusionReason:
            s += f"<br>excluded: {r.ExclusionReason}"
        out.append(s)
    return out


def lap_times_fig(laps: pd.DataFrame, a: str, b: str) -> go.Figure:
    fig = go.Figure()
    cl = laps[laps["Clean"]]
    exc = laps[(~laps["Clean"]) & laps["LapTimeS"].notna()]
    if len(cl):
        lo, hi = cl["LapTimeS"].min(), cl["LapTimeS"].max()
        exc = exc[exc["LapTimeS"] <= hi + max(1.0, 0.2 * (hi - lo))]
    fig.add_trace(go.Scatter(x=exc["LapNumber"], y=exc["LapTimeS"], mode="markers", name="excluded lap",
                             marker=dict(symbol="x", color=GREY, size=6), text=_hover(exc, True), hoverinfo="text"))
    for drv, col, sym in ((a, COL_A, "circle"), (b, COL_B, "square")):
        d = cl[cl["Driver"] == drv]
        clear, traffic = d[~d["InTraffic"]], d[d["InTraffic"]]
        fig.add_trace(go.Scatter(x=clear["LapNumber"], y=clear["LapTimeS"], mode="markers", name=f"{drv} clear air",
                                 legendgroup=drv, marker=dict(color=col, symbol=sym, size=7),
                                 text=_hover(clear), hoverinfo="text"))
        fig.add_trace(go.Scatter(x=traffic["LapNumber"], y=traffic["LapTimeS"], mode="markers",
                                 name=f"{drv} in traffic", legendgroup=drv,
                                 marker=dict(color="rgba(0,0,0,0)", symbol=sym, size=8, line=dict(color=col, width=2)),
                                 text=_hover(traffic), hoverinfo="text"))
    fig.update_xaxes(title="Lap number")
    fig.update_yaxes(title="Lap time (s)")
    return _layout(fig, "Lap times: clean laps (filled), in-traffic laps (open), excluded laps (x)")


def paired_delta_fig(paired: pd.DataFrame, pace: dict, a: str, b: str) -> go.Figure:
    fig = go.Figure()
    t = paired["EitherInTraffic"].astype(bool)
    for sel, name, kw in ((~t, "clear air (both)", dict(color="#333333")),
                          (t, "traffic (either)", dict(color="rgba(0,0,0,0)", line=dict(color=GREY, width=2)))):
        d = paired[sel]
        fig.add_trace(go.Scatter(x=d["LapNumber"], y=d["Delta_B_minus_A"], mode="markers", name=name,
                                 marker=dict(size=8, **kw),
                                 text=[f"lap {int(r.LapNumber)}<br>{r.Delta_B_minus_A:+.3f} s<br>{r.Compound_A}" for r in d.itertuples()],
                                 hoverinfo="text"))
    fig.add_hline(y=0, line_color="black", line_width=1)
    if np.isfinite(pace.get("paired_median", np.nan)):
        if np.isfinite(pace.get("paired_ci_low", np.nan)):
            fig.add_hrect(y0=pace["paired_ci_low"], y1=pace["paired_ci_high"], fillcolor=COL_A, opacity=0.12, line_width=0)
        fig.add_hline(y=pace["paired_median"], line_color=COL_A, line_width=2,
                      annotation_text=f"median {pace['paired_median']:+.3f} s (shaded: 95% bootstrap CI)",
                      annotation_position="top left")
    fig.update_xaxes(title="Lap number")
    fig.update_yaxes(title=f"{b} minus {a} (s); positive = {a} faster")
    return _layout(fig, "Same-lap-number pace delta (same compound, clean laps)")


def sector_fig(sectors: pd.DataFrame, a: str, b: str) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=sectors["Sector"], y=sectors["median_delta_B_minus_A"], name="median clean laps", marker_color="#7f7f7f")
    fig.add_bar(x=sectors["Sector"], y=sectors["best_delta_B_minus_A"], name="best sectors", marker_color="#c4c4c4")
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title=f"{b} minus {a} (s); positive = {a} faster")
    return _layout(fig, "Sector time delta: median clean laps vs best sectors", height=360)


def telemetry_fig(al: pd.DataFrame, a: str, b: str, corners: pd.DataFrame | None = None) -> go.Figure:
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[0.34, 0.2, 0.16, 0.1, 0.14])
    d = al["Distance"]
    la, lb = dict(color=COL_A, width=1.6), dict(color=COL_B, width=1.6, dash="dash")
    fig.add_trace(go.Scatter(x=d, y=al["Speed_A"], name=a, line=la, legendgroup=a), row=1, col=1)
    fig.add_trace(go.Scatter(x=d, y=al["Speed_B"], name=b, line=lb, legendgroup=b), row=1, col=1)
    fig.add_trace(go.Scatter(x=d, y=al["Delta"], name=f"delta {b}-{a}", line=dict(color="#555555", width=1.6),
                             showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=d, y=al["Throttle_A"], name=a, line=la, legendgroup=a, showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=d, y=al["Throttle_B"], name=b, line=lb, legendgroup=b, showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=d, y=al["Brake_A"].astype(float), name=f"{a} brake", line=dict(color=COL_A, width=0.5, shape="hv"),
                             fill="tozeroy", fillcolor=FILL_A, legendgroup=a, showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=d, y=-al["Brake_B"].astype(float), name=f"{b} brake", line=dict(color=COL_B, width=0.5, shape="hv"),
                             fill="tozeroy", fillcolor=FILL_B, legendgroup=b, showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=d, y=al["nGear_A"], name=a, line=dict(color=COL_A, width=1.4, shape="hv"),
                             legendgroup=a, showlegend=False), row=5, col=1)
    fig.add_trace(go.Scatter(x=d, y=al["nGear_B"], name=b, line=dict(color=COL_B, width=1.4, dash="dash", shape="hv"),
                             legendgroup=b, showlegend=False), row=5, col=1)
    if corners is not None and len(corners):
        # `corners` is the corner TABLE (columns Corner, Distance_m), as stored in the bundle
        for x in corners["Distance_m"]:
            fig.add_vline(x=float(x), line_width=0.7, line_dash="dot", line_color="#999999", row="all", col=1)
        ymin = float(min(al["Speed_A"].min(), al["Speed_B"].min()))
        fig.add_trace(go.Scatter(x=corners["Distance_m"], y=[ymin - 14] * len(corners), mode="text",
                                 text=[str(c) for c in corners["Corner"]], showlegend=False, hoverinfo="skip",
                                 textfont=dict(size=10, color="#777777")), row=1, col=1)
    fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
    fig.update_yaxes(title_text=f"Delta (s)<br>rising = {a} gaining", row=2, col=1)
    fig.update_yaxes(title_text="Throttle (%)", row=3, col=1)
    fig.update_yaxes(tickvals=[-0.5, 0.5], ticktext=[b, a], range=[-1.1, 1.1], title_text="Brake on", row=4, col=1)
    fig.update_yaxes(title_text="Gear", row=5, col=1)
    fig.update_xaxes(title_text="Distance along lap (m; B rescaled to A's length)", row=5, col=1)
    return _layout(fig, "Speed, time delta, throttle, brake and gear vs distance", height=860, hovermode="x unified")


def minisector_fig(minis: pd.DataFrame, a: str, b: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=minis["Segment"], y=minis["A_gain_s"], marker_color=[COL_A if g > 0 else COL_B for g in minis["A_gain_s"]],
        text=[f"{r.StartM:.0f}-{r.EndM:.0f} m<br>{r.A_gain_s:+.3f} s" for r in minis.itertuples()], hoverinfo="text", textposition="none"))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_xaxes(title="Mini-sector (equal distance)")
    fig.update_yaxes(title=f"Time gained by {a} (s)")
    return _layout(fig, f"Time gained per mini-sector: {a} (blue, +) vs {b} (orange, -)", height=340)


def dominance_fig(al: pd.DataFrame, minis: pd.DataFrame, a: str, b: str) -> go.Figure | None:
    if "X" not in al or al["X"].isna().all():
        return None
    fig = go.Figure()
    names = {"A": f"{a} gaining", "B": f"{b} gaining", "=": "within 5 ms"}
    colors = {"A": COL_A, "B": COL_B, "=": "#bbbbbb"}
    shown = set()
    for m in minis.itertuples():
        sel = (al["Distance"] >= m.StartM) & (al["Distance"] <= m.EndM)
        if sel.sum() < 2:
            continue
        fig.add_trace(go.Scatter(x=al.loc[sel, "X"], y=al.loc[sel, "Y"], mode="lines", line=dict(color=colors[m.Faster], width=7),
                                 name=names[m.Faster], legendgroup=m.Faster, showlegend=m.Faster not in shown,
                                 hovertemplate=f"mini-sector {m.Segment}<br>{m.A_gain_s:+.3f} s ({a} gain)<extra></extra>"))
        shown.add(m.Faster)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return _layout(fig, "Where time is gained, by mini-sector (track map)", height=520)


def stints_fig(laps: pd.DataFrame, stints: pd.DataFrame, drivers: list[str]) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(drivers), shared_yaxes=True, subplot_titles=[f"{d}" for d in drivers])
    cl = laps[laps["Clean"]]
    lo, hi = cl["LapTimeS"].min(), cl["LapTimeS"].max()
    for i, drv in enumerate(drivers, start=1):
        d_all = laps[laps["Driver"] == drv]
        exc = d_all[(~d_all["Clean"]) & d_all["LapTimeS"].notna() & (d_all["LapTimeS"] <= hi + max(1.0, 0.2 * (hi - lo)))]
        fig.add_trace(go.Scatter(x=exc["LapNumber"], y=exc["LapTimeS"], mode="markers", name="excluded lap",
                                 legendgroup="exc", showlegend=(i == 1), marker=dict(symbol="x", color=GREY, size=6),
                                 text=_hover(exc, True), hoverinfo="text"), row=1, col=i)
        d = cl[cl["Driver"] == drv]
        for comp, g in d.groupby("Compound"):
            fig.add_trace(go.Scatter(
                x=g["LapNumber"], y=g["LapTimeS"], mode="markers", name=comp.title(), legendgroup=comp,
                showlegend=(i == 1), text=_hover(g),
                hoverinfo="text", marker=dict(color=COMPOUND_COL.get(comp, "#999999"), size=8,
                                              line=dict(color="black", width=[1.5 if t else 0 for t in g["InTraffic"]]))),
                row=1, col=i)
        for s in stints[stints["Driver"] == drv].itertuples():
            seg = d[d["Stint"] == s.Stint]
            if seg.empty or not np.isfinite(s.slope_raw):
                continue
            xs = np.array([seg["LapNumber"].min(), seg["LapNumber"].max()], dtype=float)
            ys = seg["LapTimeS"].median() + s.slope_raw * (xs - seg["LapNumber"].median())
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="black", width=1.5), showlegend=False,
                                     hovertemplate=f"stint {s.Stint}: {s.slope_raw:+.3f} +/- {s.slope_raw_se:.3f} s/lap<extra></extra>"),
                          row=1, col=i)
        fig.update_xaxes(title_text="Lap number", row=1, col=i)
    fig.update_yaxes(title_text="Lap time (s)", row=1, col=1)
    return _layout(fig, "Clean lap times by stint (black ring = in traffic; line = OLS trend, apparent not isolated)", height=440)


def relpace_fig(laps: pd.DataFrame, a: str, b: str) -> go.Figure:
    fig = go.Figure()
    cl = laps[laps["Clean"] & laps["RelPaceS"].notna()]
    for drv, col, sym in ((a, COL_A, "circle"), (b, COL_B, "square")):
        d = cl[cl["Driver"] == drv]
        fig.add_trace(go.Scatter(x=d["LapNumber"], y=d["RelPaceS"], mode="markers", name=drv,
                                 marker=dict(color=col, symbol=sym, size=7), text=_hover(d), hoverinfo="text"))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_xaxes(title="Lap number")
    fig.update_yaxes(title="Lap time minus field median (s)")
    return _layout(fig, "Pace relative to the field median on the same lap", height=380)


def fuel_fig(sens: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for (drv, stint), g in sens.groupby(["Driver", "Stint"]):
        fig.add_trace(go.Scatter(x=g["assumed_fuel_effect_s_per_lap"], y=g["slope"], mode="lines+markers",
                                 name=f"{drv} stint {int(stint)} ({g['Compound'].iat[0].title()})"))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_xaxes(title="Assumed fuel-burn benefit (s/lap) - an assumption, not measured")
    fig.update_yaxes(title="Residual lap-time slope (s/lap)")
    return _layout(fig, "Sensitivity of the apparent per-lap trend to the assumed fuel effect", height=380)
