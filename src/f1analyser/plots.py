"""Matplotlib figures. Fixed A/B colours (not team colours) so teammates stay distinguishable.

Palette: Okabe-Ito blue / vermillion, plus distinct markers (identity never relies on colour alone).
Titles describe WHAT is plotted; interpretation belongs in the report, not on the figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

COL_A, COL_B = "#0072B2", "#D55E00"
MARK_A, MARK_B = "o", "s"
COMPOUND_COL = {"SOFT": "#D62728", "MEDIUM": "#E6B800", "HARD": "#555555",
                "INTERMEDIATE": "#2CA02C", "WET": "#1F77B4", "UNKNOWN": "#999999"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "legend.frameon": False, "font.size": 9.5,
})


def _save(fig, path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _style(drv, a):
    return (COL_A, MARK_A) if drv == a else (COL_B, MARK_B)


def plot_lap_distribution(clean, a, b, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    rng = np.random.default_rng(0)
    for i, drv in enumerate((a, b)):
        t = clean.loc[clean["Driver"] == drv, "LapTimeS"].dropna()
        col, mk = _style(drv, a)
        ax.boxplot(t, positions=[i], widths=0.35, showfliers=False, medianprops=dict(color="black"))
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(t)), t, s=14, color=col, marker=mk, alpha=0.7)
    ax.set_xticks([0, 1], [f"{a} (A)", f"{b} (B)"])
    ax.set_ylabel("Clean lap time (s)")
    ax.set_title("Clean-lap time distribution (box = IQR, dots = individual laps)")
    return _save(fig, path)


def plot_sector_deltas(sectors, a, b, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(sectors))
    ax.bar(x - 0.18, sectors["median_delta_B_minus_A"], 0.36, color="#7f7f7f", label="Median-lap sector delta")
    ax.bar(x + 0.18, sectors["best_delta_B_minus_A"], 0.36, color="#bbbbbb", label="Best-sector delta")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, sectors["Sector"])
    ax.set_ylabel(f"{b} minus {a} (s)   [positive = {a} faster]")
    ax.set_title("Sector time delta: median clean laps vs best sectors")
    ax.legend()
    return _save(fig, path)


def plot_paired_delta(paired, summary, a, b, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    traffic = paired["EitherInTraffic"]
    ax.scatter(paired.loc[~traffic, "LapNumber"], paired.loc[~traffic, "Delta_B_minus_A"], color="#333333", s=22,
               label="clear air (both)")
    ax.scatter(paired.loc[traffic, "LapNumber"], paired.loc[traffic, "Delta_B_minus_A"], facecolors="none",
               edgecolors="#999999", s=26, label="traffic (either)")
    ax.axhline(0, color="black", lw=0.8)
    if np.isfinite(summary.get("paired_median", np.nan)):
        ax.axhline(summary["paired_median"], color=COL_A, lw=1.5, label=f"median {summary['paired_median']:+.3f} s")
        ax.axhspan(summary["paired_ci_low"], summary["paired_ci_high"], color=COL_A, alpha=0.12, label="95% bootstrap CI")
    ax.set_xlabel("Lap number")
    ax.set_ylabel(f"{b} minus {a} (s)   [positive = {a} faster]")
    ax.set_title("Same-lap-number pace delta (same compound, clean laps)")
    ax.legend(ncol=2)
    return _save(fig, path)


def plot_telemetry(al, a, b, corners, path, lap_info: str = ""):
    fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 1.6, 1.2, 0.8, 1]))
    d = al["Distance"]
    axes[0].plot(d, al["Speed_A"], color=COL_A, lw=1.4, label=a)
    axes[0].plot(d, al["Speed_B"], color=COL_B, lw=1.4, ls="--", label=b)
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].legend(loc="lower right", ncol=2)
    axes[0].set_title("Speed, time delta, throttle, brake and gear vs distance" + (f"\n{lap_info}" if lap_info else ""))
    axes[1].plot(d, al["Delta"], color="#333333", lw=1.4)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_ylabel(f"Delta {b}-{a} (s)\nrising = {a} gaining")
    axes[2].plot(d, al["Throttle_A"], color=COL_A, lw=1.2)
    axes[2].plot(d, al["Throttle_B"], color=COL_B, lw=1.2, ls="--")
    axes[2].set_ylabel("Throttle (%)")
    axes[3].fill_between(d, 0, al["Brake_A"].astype(float), step="mid", color=COL_A, alpha=0.6)
    axes[3].fill_between(d, 0, -al["Brake_B"].astype(float), step="mid", color=COL_B, alpha=0.6)
    axes[3].set_yticks([-0.5, 0.5], [b, a])
    axes[3].set_ylabel("Brake on")
    axes[4].step(d, al["nGear_A"], color=COL_A, lw=1.2, where="post")
    axes[4].step(d, al["nGear_B"], color=COL_B, lw=1.2, ls="--", where="post")
    axes[4].set_ylabel("Gear")
    axes[4].set_xlabel("Distance along lap (m; B rescaled to A's length)")
    if corners is not None:
        for _, c in corners.iterrows():
            for ax in axes:
                ax.axvline(c["Distance"], color="#aaaaaa", lw=0.6, ls=":")
            axes[0].text(c["Distance"], axes[0].get_ylim()[0], f"{int(c['Number'])}", ha="center", va="bottom",
                         fontsize=7, color="#555555")
    return _save(fig, path)


def plot_dominance_map(al, minis, a, b, path):
    fig, ax = plt.subplots(figsize=(7, 7))
    x, y, d = al["X"].to_numpy(), al["Y"].to_numpy(), al["Distance"].to_numpy()
    segs, cols = [], []
    for _, m in minis.iterrows():
        sel = (d >= m["StartM"]) & (d <= m["EndM"])
        if sel.sum() < 2:
            continue
        segs.append(np.column_stack([x[sel], y[sel]]))
        cols.append(COL_A if m["Faster"] == "A" else (COL_B if m["Faster"] == "B" else "#bbbbbb"))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=4))
    ax.set_xlim(x.min() - 400, x.max() + 400)
    ax.set_ylim(y.min() - 400, y.max() + 400)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Where time is gained, by mini-sector\nblue = {a} gaining, orange = {b} gaining, grey = within 5 ms")
    return _save(fig, path)


def plot_minisector_bars(minis, a, b, path):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    cols = [COL_A if g > 0 else COL_B for g in minis["A_gain_s"]]
    ax.bar(minis["Segment"], minis["A_gain_s"], color=cols)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Mini-sector (equal distance)")
    ax.set_ylabel(f"Time gained by {a} (s)")
    ax.set_title(f"Per-mini-sector time gained by {a} (+) or by {b} (-)")
    return _save(fig, path)


def plot_stints(clean, all_laps, drivers, a, stints, path):
    fig, axes = plt.subplots(1, len(drivers), figsize=(6.2 * len(drivers), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, drv in zip(axes, drivers):
        d_all = all_laps[all_laps["Driver"] == drv]
        excl = d_all[~d_all["Clean"] & d_all["LapTimeS"].notna()]
        ax.scatter(excl["LapNumber"], excl["LapTimeS"], marker="x", color="#cccccc", s=14, label="excluded lap")
        d = clean[clean["Driver"] == drv]
        for comp, g in d.groupby("Compound"):
            ax.scatter(g["LapNumber"], g["LapTimeS"], color=COMPOUND_COL.get(comp, "#999"), s=20,
                       edgecolors=np.where(g["InTraffic"].fillna(False), "black", "none"), linewidths=0.8,
                       label=comp.title())
        for _, s in stints[stints["Driver"] == drv].iterrows():
            seg = d[d["Stint"] == s["Stint"]]
            xs = np.array([seg["LapNumber"].min(), seg["LapNumber"].max()])
            ys = seg["LapTimeS"].median() + s["slope_raw"] * (xs - seg["LapNumber"].median())
            ax.plot(xs, ys, color="black", lw=1.2)
        ax.set_title(f"{drv}: clean lap times by stint (black ring = in traffic; line = OLS trend)")
        ax.set_xlabel("Lap number")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Lap time (s)")
    return _save(fig, path)


def plot_relative_pace(clean, drivers, a, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    for drv in drivers:
        d = clean[clean["Driver"] == drv].dropna(subset=["RelPaceS"])
        col, mk = _style(drv, a)
        ax.scatter(d["LapNumber"], d["RelPaceS"], color=col, marker=mk, s=18, alpha=0.8, label=drv)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Lap number")
    ax.set_ylabel("Lap time minus field median (s)")
    ax.set_title("Pace relative to the field median on the same lap (removes common track/fuel trends only)")
    ax.legend()
    return _save(fig, path)


def plot_fuel_sensitivity(sens, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for (drv, stint), g in sens.groupby(["Driver", "Stint"]):
        ax.plot(g["assumed_fuel_effect_s_per_lap"], g["slope"], marker="o", label=f"{drv} stint {stint} ({g['Compound'].iat[0].title()})")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Assumed fuel-burn benefit (s/lap)  [assumption, not measured]")
    ax.set_ylabel("Residual lap-time slope (s/lap)")
    ax.set_title("Sensitivity of the apparent per-lap trend to the assumed fuel effect")
    ax.legend(fontsize=8)
    return _save(fig, path)
