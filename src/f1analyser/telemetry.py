"""Distance-aligned telemetry comparison for two laps.

Convention: Delta(d) = t_B(d) - t_A(d). Delta > 0 means A is ahead at distance d;
a RISING Delta means A is gaining on B in that stretch.

FastF1 channel notes: Speed km/h, Throttle 0-100 %, Brake is BOOLEAN (on/off, no pressure),
nGear integer. Car data is sampled ~4-5 Hz and merged with position data, so features such
as brake point have an uncertainty of roughly one sample (~15-20 m at 300 km/h).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- lap selection
def select_comparison_laps(clean: pd.DataFrame, a: str, b: str, mode: str = "matched",
                           explicit: tuple | None = None) -> dict:
    """Return {'lap_a','lap_b','mode','reason'}. Never silently falls back."""
    if explicit:
        return dict(lap_a=int(explicit[0]), lap_b=int(explicit[1]), mode="explicit",
                    reason="lap numbers supplied by the user")
    da, db = clean[clean["Driver"] == a], clean[clean["Driver"] == b]
    fastest = dict(lap_a=int(da.loc[da["LapTimeS"].idxmin(), "LapNumber"]),
                   lap_b=int(db.loc[db["LapTimeS"].idxmin(), "LapNumber"]), mode="fastest",
                   reason="each driver's fastest clean lap (tyre age / fuel may differ)")
    if mode == "fastest":
        return fastest
    cand = da.assign(_k=1).merge(db.assign(_k=1), on="_k", suffixes=("_A", "_B"))
    cand = cand[(cand["Compound_A"] == cand["Compound_B"])
                & (~cand["InTraffic_A"].fillna(False).astype(bool))
                & (~cand["InTraffic_B"].fillna(False).astype(bool))
                & ((cand["TyreAge_A"] - cand["TyreAge_B"]).abs() <= 3)
                & ((cand["LapNumber_A"] - cand["LapNumber_B"]).abs() <= 5)]
    if cand.empty:
        fastest["reason"] += " [fallback: no clear-air, same-compound, similar-age pair found]"
        return fastest
    best = cand.loc[(cand["LapTimeS_A"] + cand["LapTimeS_B"]).idxmin()]
    return dict(lap_a=int(best["LapNumber_A"]), lap_b=int(best["LapNumber_B"]), mode="matched",
                reason=(f"fastest pair on the same compound ({best['Compound_A']}), both in clear air, "
                        f"tyre age within 3 laps, lap numbers within 5"))


def get_lap_telemetry(session, driver: str, lap_number: int) -> pd.DataFrame:
    """Telemetry of one lap (FastF1 merged car+position data, with Distance)."""
    laps = session.laps
    sel = laps[(laps["Driver"] == driver) & (laps["LapNumber"] == lap_number)]
    if sel.empty:
        raise ValueError(f"No lap {lap_number} for {driver}")
    tel = sel.iloc[0].get_telemetry().copy()
    tel["T"] = (tel["Time"] - tel["Time"].iloc[0]).dt.total_seconds()
    return tel


# ----------------------------------------------------------------------------- resampling
def to_distance_grid(tel: pd.DataFrame, step_m: float = 5.0, length_m: float | None = None) -> pd.DataFrame:
    """Resample one lap onto a regular distance grid (0..length). Distance is rescaled to length_m."""
    d = tel["Distance"].to_numpy(float)
    t = tel["T"].to_numpy(float)
    keep = np.r_[True, np.diff(d) > 0]            # strictly increasing distance only
    d, t = d[keep], t[keep]
    own_len = float(d[-1])
    scale = (length_m / own_len) if length_m else 1.0
    d = d * scale
    grid = np.arange(0.0, d[-1], step_m)

    def lin(col):
        v = tel[col].to_numpy(float)[keep]
        return np.interp(grid, d, v)

    out = pd.DataFrame({"Distance": grid, "T": np.interp(grid, d, t)})
    for col in ("Speed", "Throttle", "X", "Y", "RPM"):
        if col in tel:
            out[col] = lin(col)
    # step channels: take the previous sample (no interpolation between gears / brake states)
    idx = np.clip(np.searchsorted(d, grid, side="right") - 1, 0, len(d) - 1)
    out["Brake"] = tel["Brake"].astype(float).to_numpy()[keep][idx] > 0.5
    out["nGear"] = tel["nGear"].to_numpy(float)[keep][idx]
    out.attrs["length_m"] = own_len
    out.attrs["scale"] = scale
    return out


def path_distance(tel_a: pd.DataFrame, tel_b: pd.DataFrame, window_back_m: float = 30.0, window_fwd_m: float = 250.0,
                  smooth_s: float = 6.0):
    """Distance of every B sample along A's lap, found by projecting B's (X, Y) onto A's path.

    Integrating integer km/h speed drifts by tens of metres between two cars, which smears the delta-time curve.
    The track position does not drift, so B's distance is taken from where B is on A's trajectory.
    Tracking is sequential and windowed (B can only move forward), so the crossing of a figure-eight track
    (Suzuka) cannot snap B onto the wrong branch. Returns None if X/Y are missing."""
    for tel in (tel_a, tel_b):
        if not {"X", "Y"} <= set(tel.columns) or tel[["X", "Y"]].isna().any().any():
            return None
    da = tel_a["Distance"].to_numpy(float)
    keep = np.r_[True, np.diff(da) > 0]
    da = da[keep]
    xa, ya = tel_a["X"].to_numpy(float)[keep], tel_a["Y"].to_numpy(float)[keep]
    if len(da) < 10:
        return None
    # densify A's path (about 40 points per sample gap) with distance interpolated along it
    k = 40
    frac = np.linspace(0, 1, k, endpoint=False)
    idx = np.arange(len(da) - 1)
    dd = (da[idx, None] + (da[idx + 1] - da[idx])[:, None] * frac).ravel()
    xx = (xa[idx, None] + (xa[idx + 1] - xa[idx])[:, None] * frac).ravel()
    yy = (ya[idx, None] + (ya[idx + 1] - ya[idx])[:, None] * frac).ravel()
    xb, yb = tel_b["X"].to_numpy(float), tel_b["Y"].to_numpy(float)
    out = np.empty(len(xb))
    prev = 0.0
    for j in range(len(xb)):
        lo, hi = np.searchsorted(dd, [prev - window_back_m, prev + window_fwd_m])
        hi = max(hi, lo + 1)
        i = lo + int(np.argmin((xx[lo:hi] - xb[j]) ** 2 + (yy[lo:hi] - yb[j]) ** 2))
        prev = max(prev, float(dd[min(i, len(dd) - 1)]))
        out[j] = prev
    # Complementary filter: short-term shape from B's own speed integral (smooth, accurate over seconds), long-term
    # level from the track position (no drift). Raw projections of 3.7 Hz positions are steppy; they would make
    # the delta curve jump, so only their slowly varying difference to the speed integral is used.
    t = tel_b["T"].to_numpy(float)
    db = tel_b["Distance"].to_numpy(float)
    dt = float(np.median(np.diff(t))) if len(t) > 3 else 0.0
    if dt > 0:
        from scipy.ndimage import gaussian_filter1d
        err = gaussian_filter1d(out - db, sigma=max(1.0, smooth_s / dt), mode="nearest")
        out = db + err
        out = np.maximum.accumulate(out)
    return out


def align_laps(tel_a: pd.DataFrame, tel_b: pd.DataFrame, step_m: float = 5.0):
    """Common grid on A's length. B's distance comes from its track position on A's path when X/Y exist, otherwise
    from integrated speed rescaled to A's length. Returns (aligned, info)."""
    method = "speed integral (rescaled)"
    n_stale = int(tel_a.attrs.get("stale_speed_samples_dropped", 0)) + int(tel_b.attrs.get("stale_speed_samples_dropped", 0))
    pd_b = path_distance(tel_a, tel_b)
    if pd_b is not None and np.isfinite(pd_b).all() and pd_b[-1] > 0.9 * float(tel_a["Distance"].iloc[-1]):
        tel_b = tel_b.copy()
        tel_b["Distance"] = pd_b
        method = "track position (X/Y) projected on A's path"
    ga = to_distance_grid(tel_a, step_m)
    L = ga.attrs["length_m"]
    gb = to_distance_grid(tel_b, step_m, length_m=L)
    n = min(len(ga), len(gb))
    ga, gb = ga.iloc[:n].reset_index(drop=True), gb.iloc[:n].reset_index(drop=True)
    al = pd.DataFrame({"Distance": ga["Distance"], "Delta": gb["T"] - ga["T"]})
    for col in ("Speed", "Throttle", "Brake", "nGear", "T"):
        al[f"{col}_A"], al[f"{col}_B"] = ga[col], gb[col]
    al["X"], al["Y"] = ga.get("X"), ga.get("Y")
    info = dict(length_a_m=L, length_b_m=gb.attrs["length_m"], b_distance_scale=gb.attrs["scale"],
                lap_time_a_s=float(tel_a["T"].iloc[-1]), lap_time_b_s=float(tel_b["T"].iloc[-1]),
                alignment=method, stale_speed_samples_dropped=n_stale)
    dist, dl = al["Distance"].to_numpy(), al["Delta"].to_numpy()
    if len(al) > 12 and dist[-1] > 100:
        g = np.arange(0.0, dist[-1] - 50.0, step_m)
        jump = np.abs(np.interp(g + 50.0, dist, dl) - np.interp(g, dist, dl))
        ta = al["T_A"].to_numpy()
        win_t = np.maximum(np.interp(g + 50.0, dist, ta) - np.interp(g, dist, ta), 1e-6)
        info["max_delta_jump_50m_s"] = float(jump.max())
        # The same jump as a share of the time A needs for those 50 m = the speed difference the curve implies.
        # Windows where A's speed changes by more than 25 % (braking, acceleration) are skipped: there a 10-20 m
        # position or sampling error alone makes the two speeds differ a lot, so a jump cannot be judged.
        spd = al["Speed_A"].to_numpy(float)
        sp_lo, sp_hi = np.interp(g, dist, spd), np.interp(g + 50.0, dist, spd)
        steady = np.abs(sp_hi - sp_lo) / np.maximum((sp_hi + sp_lo) / 2, 1.0) < 0.25
        frac = jump / win_t
        info["max_implied_speed_diff_frac"] = float(frac[steady].max()) if steady.any() else 0.0
    return al, info


# ----------------------------------------------------------------------------- features
def minisector_deltas(al: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Time gained by A over B (positive) or lost (negative) in each equal-length mini-sector."""
    L = float(al["Distance"].iloc[-1])
    edges = np.linspace(0, L, n + 1)
    delta_at = np.interp(edges, al["Distance"], al["Delta"])
    rows = []
    for i in range(n):
        mid = (edges[i] + edges[i + 1]) / 2
        gain = float(delta_at[i + 1] - delta_at[i])
        rows.append(dict(Segment=i + 1, StartM=edges[i], EndM=edges[i + 1], A_gain_s=gain,
                         Faster="A" if gain > 0.005 else ("B" if gain < -0.005 else "=")))
    return pd.DataFrame(rows)


def normalise_corners(corners: pd.DataFrame | None, lap_length_m: float) -> pd.DataFrame | None:
    """Defensive unit check: if corner distances look like decimetres, convert to metres."""
    if corners is None or corners.empty:
        return None
    c = corners.copy()
    if c["Distance"].max() > 1.5 * lap_length_m:
        c["Distance"] = c["Distance"] / 10.0
    return c[(c["Distance"] > 0) & (c["Distance"] < lap_length_m)].reset_index(drop=True)


def _first_true(distance: np.ndarray, mask: np.ndarray) -> float:
    i = np.flatnonzero(mask)
    return float(distance[i[0]]) if i.size else float("nan")


def corner_table(al: pd.DataFrame, corners: pd.DataFrame, a: str, b: str,
                 before: float = 150.0, after: float = 150.0, full_throttle: float = 95.0) -> pd.DataFrame:
    """Per corner: brake point, minimum speed, full-throttle point, and time gained in entry / exit.

    Entry = [corner-before, corner]; exit = [corner, corner+after]. Gain > 0: A gained on B.
    Brake point is the first False->True brake transition inside the entry window (NaN if the
    brake is already on at the window start, e.g. closely spaced corners).
    """
    rows = []
    d = al["Distance"].to_numpy()
    for _, c in corners.iterrows():
        cd = float(c["Distance"])
        lo, hi = max(cd - before, 0.0), min(cd + after, d[-1])
        w = (d >= lo) & (d <= hi)
        if w.sum() < 5:
            continue
        row = dict(Corner=f"{int(c['Number'])}{c.get('Letter', '') or ''}", Distance_m=cd)
        for tag, drv in (("A", a), ("B", b)):
            sp, br, th = al[f"Speed_{tag}"].to_numpy(), al[f"Brake_{tag}"].to_numpy(), al[f"Throttle_{tag}"].to_numpy()
            pre = (d >= lo) & (d <= cd)
            rising = np.r_[False, (~br[:-1]) & br[1:]]
            row[f"BrakePt_{drv}_m"] = float("nan") if br[pre][0] else _first_true(d, pre & rising)
            iw = np.flatnonzero(w)
            imin = iw[np.argmin(sp[iw])]
            row[f"MinSpeed_{drv}_kmh"] = float(sp[imin])
            row[f"MinSpeedAt_{drv}_m"] = float(d[imin])
            post = (d >= d[imin]) & (d <= hi)
            row[f"FullThrottle_{drv}_m"] = _first_true(d, post & (th >= full_throttle))
        dl = lambda x: float(np.interp(x, d, al["Delta"]))
        row["EntryGain_A_s"] = dl(cd) - dl(lo)
        row["ExitGain_A_s"] = dl(hi) - dl(cd)
        row["A_brakes_later_m"] = row[f"BrakePt_{a}_m"] - row[f"BrakePt_{b}_m"]
        row["A_min_speed_higher_kmh"] = row[f"MinSpeed_{a}_kmh"] - row[f"MinSpeed_{b}_kmh"]
        row["A_full_throttle_earlier_m"] = row[f"FullThrottle_{b}_m"] - row[f"FullThrottle_{a}_m"]
        rows.append(row)
    return pd.DataFrame(rows)


def channel_summary(al: pd.DataFrame, a: str, b: str, full_throttle: float = 95.0) -> pd.DataFrame:
    """Lap-level channel summary (time-weighted). Descriptive only; setup inference needs more data."""
    dt = np.gradient(al["T_A"].to_numpy())
    rows = []
    for tag, drv in (("A", a), ("B", b)):
        dt = np.gradient(al[f"T_{tag}"].to_numpy())
        w = dt / dt.sum()
        rows.append(dict(Driver=drv, top_speed_kmh=float(al[f"Speed_{tag}"].max()),
                         min_speed_kmh=float(al[f"Speed_{tag}"].min()),
                         pct_time_full_throttle=float(100 * (w * (al[f"Throttle_{tag}"] >= full_throttle)).sum()),
                         pct_time_braking=float(100 * (w * al[f"Brake_{tag}"].astype(float)).sum()),
                         gear_changes=int((np.diff(al[f"nGear_{tag}"].to_numpy()) != 0).sum())))
    return pd.DataFrame(rows)
