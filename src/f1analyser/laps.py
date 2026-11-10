"""Lap cleaning, best laps, sector comparison, consistency and paired-lap comparison.

Sign convention used everywhere in this project:
    delta = time_B - time_A      ->  delta > 0 means driver A is FASTER.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from . import stats as st

SECTORS = ("S1", "S2", "S3")


def _secs(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def prepare_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Plain DataFrame copy with numeric second columns (LapTimeS, S1S..S3S, TimeS)."""
    df = pd.DataFrame(laps).copy()
    df["LapTimeS"] = _secs(df["LapTime"])
    for i, name in enumerate(("Sector1Time", "Sector2Time", "Sector3Time"), start=1):
        df[f"S{i}S"] = _secs(df[name]) if name in df else np.nan
    df["TimeS"] = _secs(df["Time"]) if "Time" in df else np.nan
    for col, default in (("Stint", np.nan), ("Compound", "UNKNOWN"), ("TyreLife", np.nan),
                         ("TrackStatus", ""), ("Deleted", False), ("IsAccurate", True)):
        if col not in df:
            df[col] = default
    df["Compound"] = df["Compound"].fillna("UNKNOWN").astype(str)
    return df


def flag_traffic(df: pd.DataFrame, threshold_s: float) -> pd.DataFrame:
    """Approximate traffic from lap-end timing: gap to the car that crossed the line just ahead.

    A lap is flagged if the gap at the end of this lap OR the previous lap is < threshold.
    This is a proxy (one sample per lap at the timing line), not a measured gap through the lap.
    """
    df = df.copy()
    d = df[["Driver", "LapNumber", "TimeS"]].dropna().sort_values(["LapNumber", "TimeS"])
    d["GapAheadS"] = d.groupby("LapNumber")["TimeS"].diff()
    df = df.drop(columns=["GapAheadS"], errors="ignore").merge(
        d[["Driver", "LapNumber", "GapAheadS"]], on=["Driver", "LapNumber"], how="left")
    df = df.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    prev = df.groupby("Driver")["GapAheadS"].shift(1)
    df["MinGapS"] = np.fmin(df["GapAheadS"], prev)
    df["InTraffic"] = df["MinGapS"] < threshold_s
    return df


def flag_clean(df: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    """Add ``Clean`` (bool) and ``ExclusionReason`` (first matching reason, for the audit table)."""
    df = df.copy()
    allowed = {"1", "2"} if cfg.allow_yellow else {"1"}

    def status_ok(ts) -> bool:
        s = "" if pd.isna(ts) else str(ts)
        return s != "" and set(s) <= allowed

    reason = pd.Series("", index=df.index, dtype=object)

    def mark(mask, label):
        nonlocal reason
        reason = reason.mask((reason == "") & mask, label)

    mark(df["LapTimeS"].isna(), "no_lap_time")
    mark(df["LapNumber"] == 1, "lap_1")
    pit = pd.Series(False, index=df.index)
    for col in ("PitInTime", "PitOutTime"):
        if col in df:
            pit |= df[col].notna()
    mark(pit, "in_or_out_lap")
    mark(~df["TrackStatus"].apply(status_ok), "not_green")
    mark(df["Deleted"].fillna(False).astype(bool), "deleted_track_limits")
    if cfg.use_is_accurate:
        mark(~df["IsAccurate"].fillna(False).astype(bool), "not_accurate")

    # slow-lap filter relative to each driver's best otherwise-valid lap
    ok = reason == ""
    best = df.loc[ok].groupby("Driver")["LapTimeS"].min()
    limit = df["Driver"].map(best) * cfg.quick_lap_threshold
    mark(ok & (df["LapTimeS"] > limit), "slow_lap_gt_threshold")

    df["ExclusionReason"] = reason
    df["Clean"] = reason == ""
    return df


def exclusion_summary(df: pd.DataFrame, drivers=None) -> pd.DataFrame:
    d = df if drivers is None else df[df["Driver"].isin(drivers)]
    t = d.assign(Reason=d["ExclusionReason"].replace("", "kept (clean)")) \
        .pivot_table(index="Driver", columns="Reason", values="LapNumber", aggfunc="count", fill_value=0)
    return t.reset_index()


def best_laps(clean: pd.DataFrame, drivers) -> pd.DataFrame:
    rows = []
    for drv in drivers:
        d = clean[clean["Driver"] == drv]
        if d.empty:
            continue
        r = d.loc[d["LapTimeS"].idxmin()]
        rows.append(dict(Driver=drv, BestLapS=r["LapTimeS"], LapNumber=int(r["LapNumber"]),
                         Compound=r["Compound"], TyreLife=r["TyreLife"],
                         S1S=r["S1S"], S2S=r["S2S"], S3S=r["S3S"], InTraffic=bool(r.get("InTraffic", False))))
    return pd.DataFrame(rows)


def lap_stats(clean: pd.DataFrame, drivers) -> pd.DataFrame:
    rows = []
    for drv in drivers:
        t = clean.loc[clean["Driver"] == drv, "LapTimeS"].dropna().sort_values()
        if t.empty:
            continue
        rows.append(dict(
            Driver=drv, n_clean=len(t), best=t.iloc[0], top5_mean=t.iloc[:5].mean(),
            median=t.median(), mean=t.mean(), std=t.std(), mad=st.mad(t),
            iqr=t.quantile(0.75) - t.quantile(0.25), p90=t.quantile(0.90),
        ))
    return pd.DataFrame(rows)


def sector_table(clean: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Best and median sector times per driver; delta = B - A (positive: A faster)."""
    rows = []
    for s in SECTORS:
        col = f"{s}S"
        ta = clean.loc[clean["Driver"] == a, col].dropna()
        tb = clean.loc[clean["Driver"] == b, col].dropna()
        rows.append({
            "Sector": s,
            f"best_{a}": ta.min(), f"best_{b}": tb.min(), "best_delta_B_minus_A": tb.min() - ta.min(),
            f"median_{a}": ta.median(), f"median_{b}": tb.median(),
            "median_delta_B_minus_A": tb.median() - ta.median(),
            f"n_{a}": len(ta), f"n_{b}": len(tb),
        })
    out = pd.DataFrame(rows)
    tot = out["median_delta_B_minus_A"].sum()
    out["share_of_median_gap"] = out["median_delta_B_minus_A"] / tot if tot else np.nan
    return out


def theoretical_best(clean: pd.DataFrame, drivers) -> pd.DataFrame:
    rows = []
    for drv in drivers:
        d = clean[clean["Driver"] == drv]
        bs = [d[f"{s}S"].min() for s in SECTORS]
        actual = d["LapTimeS"].min()
        rows.append(dict(Driver=drv, theoretical_best=float(np.nansum(bs)), actual_best=actual,
                         gap_to_theoretical=actual - float(np.nansum(bs))))
    return pd.DataFrame(rows)


def paired_lap_delta(clean: pd.DataFrame, a: str, b: str, same_compound: bool = True) -> pd.DataFrame:
    """Match laps by lap number (similar fuel, track state, same race moment)."""
    cols = ["LapNumber", "LapTimeS", "Compound", "TyreLife", "InTraffic"]
    da = clean.loc[clean["Driver"] == a, cols]
    db = clean.loc[clean["Driver"] == b, cols]
    m = da.merge(db, on="LapNumber", suffixes=("_A", "_B"))
    if same_compound:
        m = m[m["Compound_A"] == m["Compound_B"]]
    m["Delta_B_minus_A"] = m["LapTimeS_B"] - m["LapTimeS_A"]
    m["TyreLifeDiff_B_minus_A"] = m["TyreLife_B"] - m["TyreLife_A"]
    m["EitherInTraffic"] = m["InTraffic_A"].fillna(False).astype(bool) | m["InTraffic_B"].fillna(False).astype(bool)
    return m.sort_values("LapNumber").reset_index(drop=True)


def pace_gap_summary(paired: pd.DataFrame, clean: pd.DataFrame, a: str, b: str, cfg: AnalysisConfig) -> dict:
    """Is the gap consistent, or does it hinge on a few laps? Returns numbers for findings/report."""
    out: dict = {}
    d = paired["Delta_B_minus_A"].to_numpy()
    est, lo, hi = st.bootstrap_ci(d, np.median, cfg.n_bootstrap, seed=cfg.seed)
    share, n, p = st.sign_share(d)
    out.update(paired_n=int(len(d)), paired_median=est, paired_ci_low=lo, paired_ci_high=hi,
               share_laps_A_faster=share, sign_p=p)
    clear = paired.loc[~paired["EitherInTraffic"], "Delta_B_minus_A"].to_numpy()
    est_c, lo_c, hi_c = st.bootstrap_ci(clear, np.median, cfg.n_bootstrap, seed=cfg.seed)
    out.update(clear_n=int(len(clear)), clear_median=est_c, clear_ci_low=lo_c, clear_ci_high=hi_c)
    # leave-one-out: does removing the single most extreme lap change the sign / size?
    if len(d) >= 5:
        j = int(np.argmax(np.abs(d - np.median(d))))
        out["median_without_most_extreme_lap"] = float(np.median(np.delete(d, j)))
    ta = clean.loc[clean["Driver"] == a, "LapTimeS"].sort_values()
    tb = clean.loc[clean["Driver"] == b, "LapTimeS"].sort_values()
    if len(ta) and len(tb):
        out["best_lap_gap"] = float(tb.iloc[0] - ta.iloc[0])
        out["top5_mean_gap"] = float(tb.iloc[:5].mean() - ta.iloc[:5].mean())
        e2, l2, h2 = st.bootstrap_diff_ci(ta, tb, np.median, cfg.n_bootstrap, seed=cfg.seed)
        out.update(unpaired_median_gap=e2, unpaired_ci_low=l2, unpaired_ci_high=h2)
    return out
