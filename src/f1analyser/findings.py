"""Candidate observations with evidence, limits and the data that would be needed to go further.

These are DRAFTS. The analyst must look at the figures, keep the 3-5 that survive scrutiny and
rewrite them. The wording deliberately says "consistent with" / "cannot distinguish" - the data
available from FastF1 cannot isolate tyre degradation, braking technique or car setup on its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _f(x, nd=3, sign=True):
    return "n/a" if x is None or not np.isfinite(x) else (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}")


def build_findings(a: str, b: str, pace: dict, sectors: pd.DataFrame, stints: pd.DataFrame,
                   fuel: pd.DataFrame, minis: pd.DataFrame | None, corners: pd.DataFrame | None,
                   tel_info: dict | None, tel_laps: dict | None) -> list[dict]:
    out: list[dict] = []
    faster = a if pace.get("paired_median", 0) > 0 else b
    slower = b if faster == a else a

    # 1. overall pace gap and robustness -------------------------------------------------------
    if pace.get("paired_n", 0) >= 5:
        med = abs(pace["paired_median"])
        out.append(dict(
            id="pace_gap",
            title=f"{faster} is faster over comparable laps by a median of {med:.3f} s",
            evidence=(f"Paired by lap number (same compound, clean laps, n={pace['paired_n']}): median "
                      f"({b}-{a}) = {_f(pace['paired_median'])} s, 95% bootstrap CI "
                      f"[{_f(pace['paired_ci_low'])}, {_f(pace['paired_ci_high'])}]; {a} faster on "
                      f"{100 * pace['share_laps_A_faster']:.0f}% of laps (sign-test p={pace['sign_p']:.3g}). "
                      f"Clear-air laps only (n={pace['clear_n']}): {_f(pace['clear_median'])} s "
                      f"[{_f(pace['clear_ci_low'])}, {_f(pace['clear_ci_high'])}]."),
            robustness=(f"Dropping the single most extreme lap gives {_f(pace.get('median_without_most_extreme_lap'))} s. "
                        f"Best-lap gap {_f(pace.get('best_lap_gap'))} s vs top-5 mean gap {_f(pace.get('top5_mean_gap'))} s "
                        f"vs median gap {_f(pace['paired_median'])} s: if these differ a lot, the headline depends on "
                        f"which laps you choose."),
            caveat=("Same lap number does not guarantee same tyre age, fuel, engine mode or instruction "
                    "(e.g. lift-and-coast, managing a gap). A CI that excludes zero says the gap is "
                    "unlikely to be sampling noise, not why it exists."),
            needed="Team radio / strategy context, tyre age per lap, engine-mode data, gaps through the lap."))

    # 2. where in the lap: sectors ---------------------------------------------------------------
    if sectors is not None and len(sectors):
        top = sectors.loc[sectors["median_delta_B_minus_A"].abs().idxmax()]
        out.append(dict(
            id="sector_location",
            title=f"The largest median sector gap is in {top['Sector']}",
            evidence="; ".join(f"{r['Sector']}: {_f(r['median_delta_B_minus_A'])} s "
                               f"({100 * r['share_of_median_gap']:.0f}% of summed gap)"
                               for _, r in sectors.iterrows()) + f"  [delta = {b}-{a}, + means {a} faster].",
            robustness="Compare with best-sector deltas in the same table: one-lap potential vs race-trim pace can differ.",
            caveat="Sectors are 1-3 km long; a sector gap cannot say whether entry, apex or exit is responsible.",
            needed="Distance-aligned telemetry (below) to localise within the sector."))

    # 3. localisation from telemetry -------------------------------------------------------------
    if minis is not None and len(minis):
        m = minis.reindex(minis["A_gain_s"].abs().sort_values(ascending=False).index).head(3)
        txt = "; ".join(f"mini-sector {int(r['Segment'])} ({r['StartM']:.0f}-{r['EndM']:.0f} m): "
                        f"{_f(r['A_gain_s'])} s" for _, r in m.iterrows())
        evid = (f"On laps {tel_laps['lap_a']} ({a}) and {tel_laps['lap_b']} ({b}) [{tel_laps['mode']}: "
                f"{tel_laps['reason']}], the three largest per-mini-sector differences are {txt} "
                f"(+ = {a} gained).")
        if corners is not None and len(corners):
            c = corners.reindex((corners["EntryGain_A_s"].abs() + corners["ExitGain_A_s"].abs())
                                .sort_values(ascending=False).index).head(2)
            evid += " Corner detail: " + "; ".join(
                f"T{r['Corner']}: entry {_f(r['EntryGain_A_s'])} s, exit {_f(r['ExitGain_A_s'])} s, "
                f"min-speed diff {_f(r['A_min_speed_higher_kmh'], 1)} km/h, "
                f"brake point {_f(r['A_brakes_later_m'], 0)} m ({a} later if +)"
                for _, r in c.iterrows()) + "."
        out.append(dict(
            id="telemetry_localisation", title="Where the time difference accrues on the compared laps",
            evidence=evid,
            robustness="Repeat on a second pair of laps (--tel-laps) and check the same mini-sectors reappear.",
            caveat=("Two laps only. Telemetry is ~4-5 Hz with integrated distance, so brake points have ~15-20 m "
                    "uncertainty; Brake is on/off (no pressure), so 'braking technique' cannot be read from it."
                    + (f" ALIGNMENT WARNING: the delta curve changes by {tel_info['max_delta_jump_50m_s']:.2f} s within 50 m "
                       f"somewhere on the lap, which implies a {100 * tel_info['max_implied_speed_diff_frac']:.0f}% speed "
                       "difference over that stretch and is probably an alignment artifact; do not quote the "
                       "localisation without a second lap pair."
                       if tel_info and tel_info.get("max_implied_speed_diff_frac", 0) > 0.15 else "")),
            needed="More lap pairs; pedal pressure / steering data (not public); same-tyre-age laps."))

    # 4. stint trends -----------------------------------------------------------------------------
    if stints is not None and len(stints):
        parts = []
        for _, r in stints.iterrows():
            sig = "distinguishable from zero" if (np.isfinite(r["slope_raw_se"]) and abs(r["slope_raw"]) > 2 * r["slope_raw_se"]) \
                else "not distinguishable from zero"
            f03 = fuel[(fuel["Driver"] == r["Driver"]) & (fuel["Stint"] == r["Stint"])
                       & (np.isclose(fuel["assumed_fuel_effect_s_per_lap"], 0.03))]
            f03s = f" -> {_f(f03['slope'].iat[0])} s/lap if 0.03 s/lap fuel benefit is assumed" if len(f03) else ""
            parts.append(f"{r['Driver']} stint {r['Stint']} ({r['Compound'].title()}, laps {r['first_lap']}-{r['last_lap']}, "
                         f"n={r['n_clean']}): raw {_f(r['slope_raw'])} +/- {r['slope_raw_se']:.3f} s/lap ({sig}){f03s}; "
                         f"vs field {_f(r['slope_vs_field'])} s/lap")
        out.append(dict(
            id="stint_trend", title="Lap-time trend over each stint (apparent, not isolated tyre degradation)",
            evidence=" | ".join(parts),
            robustness="Compare raw, fuel-assumption, field-relative and clear-air slopes in stint_table.csv; "
                       "a conclusion that survives all four is stronger.",
            caveat=("Tyre wear is confounded with fuel burn, track evolution, traffic, temperature and driver "
                    "management; stints differ in compound, length and start age. The fuel effect is an assumption, "
                    "not a measurement, so these numbers are NOT tyre degradation rates."),
            needed="Fuel load / start mass, tyre temperatures and pressures, multiple stints on the same compound, "
                   "team stint plans (public data has none of these)."))

    # 5. what the data cannot say -----------------------------------------------------------------
    out.append(dict(
        id="limits", title="What this analysis cannot establish",
        evidence=("Public timing/telemetry has no fuel mass, tyre temperature/pressure, ride height, wing level, "
                  "brake pressure, steering angle or engine mode."),
        robustness="n/a",
        caveat=("Do not state that a gap is 'caused by' tyre degradation, braking technique or setup. At most: the "
                "pattern is consistent with X, and here is the extra data that would confirm or reject it."),
        needed="Teammate comparison across several races; setup sheets / sim data from the team."))
    return out
