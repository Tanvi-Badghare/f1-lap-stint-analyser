"""Stint analysis.

The slopes produced here are *apparent lap-time trends*, NOT isolated tyre degradation.
Within a stint, tyre age, fuel burn, track evolution, traffic and driver management all
move together. We therefore report: raw slope, slope under a *range of assumed* fuel
effects, slope relative to the field median, and slope on clear-air laps only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from . import stats as st


def add_tyre_age(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a usable tyre-age column: FastF1 TyreLife, else laps since stint start + 1."""
    df = df.copy()
    first = df.groupby(["Driver", "Stint"])["LapNumber"].transform("min")
    fallback = df["LapNumber"] - first + 1
    df["TyreAge"] = df["TyreLife"].where(df["TyreLife"].notna(), fallback)
    return df


def field_reference(clean: pd.DataFrame, min_cars: int = 5) -> pd.Series:
    """Median clean lap time per lap number across ALL drivers (needs >= min_cars samples)."""
    g = clean.groupby("LapNumber")["LapTimeS"]
    ref = g.median()[g.count() >= min_cars]
    return ref.rename("FieldMedianS")


def add_relative_pace(clean: pd.DataFrame, ref: pd.Series) -> pd.DataFrame:
    df = clean.copy()
    df["FieldMedianS"] = df["LapNumber"].map(ref)
    df["RelPaceS"] = df["LapTimeS"] - df["FieldMedianS"]
    return df


def stint_table(clean: pd.DataFrame, drivers, cfg: AnalysisConfig) -> pd.DataFrame:
    """One row per driver/stint with several robustness variants of the lap-time slope."""
    rows = []
    for drv in drivers:
        d_all = clean[clean["Driver"] == drv]
        for stint, s in d_all.groupby("Stint"):
            if len(s) < cfg.min_stint_laps:
                continue
            comp = s["Compound"].mode().iat[0]
            base = dict(Driver=drv, Stint=int(stint) if pd.notna(stint) else -1, Compound=comp,
                        first_lap=int(s["LapNumber"].min()), last_lap=int(s["LapNumber"].max()),
                        tyre_age_start=float(s["TyreAge"].min()), n_clean=len(s),
                        n_clear_air=int((~s["InTraffic"].fillna(False)).sum()),
                        median_lap_s=float(s["LapTimeS"].median()))
            raw = st.linear_trend(s["TyreAge"], s["LapTimeS"])
            base.update(slope_raw=raw["slope"], slope_raw_se=raw["stderr"], slope_raw_ci_low=raw["ci_low"],
                        slope_raw_ci_high=raw["ci_high"], resid_std=raw["resid_std"], r2=raw["r2"])
            clear = s[~s["InTraffic"].fillna(False)]
            c = st.linear_trend(clear["TyreAge"], clear["LapTimeS"])
            base.update(slope_clear_air=c["slope"], slope_clear_air_se=c["stderr"], n_slope_clear_air=c["n"])
            if "RelPaceS" in s and s["RelPaceS"].notna().sum() >= 3:
                r = st.linear_trend(s["TyreAge"], s["RelPaceS"])
                base.update(slope_vs_field=r["slope"], slope_vs_field_se=r["stderr"])
            else:
                base.update(slope_vs_field=np.nan, slope_vs_field_se=np.nan)
            rows.append(base)
    return pd.DataFrame(rows)


def fuel_sensitivity(clean: pd.DataFrame, drivers, cfg: AnalysisConfig) -> pd.DataFrame:
    """Slope after ADDING back an assumed fuel-burn benefit of f s/lap (f is an assumption, not data)."""
    rows = []
    for drv in drivers:
        for stint, s in clean[clean["Driver"] == drv].groupby("Stint"):
            if len(s) < cfg.min_stint_laps:
                continue
            for f in cfg.fuel_effects:
                y = s["LapTimeS"] + f * (s["LapNumber"] - 1)
                t = st.linear_trend(s["TyreAge"], y)
                rows.append(dict(Driver=drv, Stint=int(stint) if pd.notna(stint) else -1,
                                 Compound=s["Compound"].mode().iat[0], assumed_fuel_effect_s_per_lap=f,
                                 slope=t["slope"], ci_low=t["ci_low"], ci_high=t["ci_high"], n=t["n"]))
    return pd.DataFrame(rows)


def compound_comparison(stints: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Like-for-like view: same compound, both drivers, slope + SE side by side."""
    rows = []
    for comp in sorted(set(stints["Compound"])):
        for drv in (a, b):
            s = stints[(stints["Driver"] == drv) & (stints["Compound"] == comp)]
            if len(s):
                r = s.iloc[0]
                rows.append(dict(Compound=comp, Driver=drv, Stint=r["Stint"], n=r["n_clean"],
                                 slope_raw=r["slope_raw"], se=r["slope_raw_se"],
                                 slope_vs_field=r["slope_vs_field"]))
    return pd.DataFrame(rows)
