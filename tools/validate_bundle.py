"""Cross-checks a saved bundle against itself.  python validate_bundle.py web_data/2026_japan_R_PIA-vs-NOR
Each check prints OK / CHECK. CHECK means 'look at this', not necessarily a bug."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

d = Path(sys.argv[1])
b = json.loads((d / "bundle.json").read_text(encoding="utf-8"))
A, B = b["cfg"]["driver_a"], b["cfg"]["driver_b"]
laps = pd.read_csv(d / "laps.csv")
paired = pd.read_csv(d / "paired.csv")
best = pd.read_csv(d / "best_laps.csv")
tel = b.get("tel_info") or {}
tl = b.get("tel_laps") or {}


def flag(ok, msg):
    print(("OK    " if ok else "CHECK ") + msg)


print(f"{A} vs {B}  (delta = {B} - {A}; positive = {A} faster)\n")

# 1. sectors add up to the lap time
x = laps.dropna(subset=["LapTimeS", "S1S", "S2S", "S3S"])
err = (x.S1S + x.S2S + x.S3S - x.LapTimeS).abs()
flag(err.max() < 0.05, f"sector sum vs lap time: max error {err.max():.3f} s over {len(x)} laps")

# 2. paired gap statistics
dl = paired["Delta_B_minus_A"].dropna()
print(f"      paired laps n={len(dl)}, median {dl.median():+.3f}, mean {dl.mean():+.3f}, {A} faster on {(dl > 0).mean():.0%}")
cl = laps[laps.Clean]
bl = cl.groupby("Driver").LapTimeS.min()
bg = bl.get(B, np.nan) - bl.get(A, np.nan)
flag(abs(dl.median() - bg) > 0.0015, f"median paired gap {dl.median():+.3f} vs best-lap gap {bg:+.3f} (identical values would be a coincidence)")
print(f"      lap 1 / lap 2 times: " + ", ".join(f"{r.Driver} L{int(r.LapNumber)} {r.LapTimeS:.1f}" for r in laps[laps.LapNumber.isin([1, 2])].itertuples()))
slow = laps[(laps.LapTimeS > 1.07 * laps.groupby("Driver").LapTimeS.transform("min")) & (laps.LapTimeS.notna())]
print(f"      laps > 107% of own best: {sorted(slow.LapNumber.unique().tolist())}  (pit, SC/VSC or incidents; the lap chart axis may hide them)")

# 3. telemetry
if tel and (d / "aligned.csv").exists() and (d / "aligned.csv").stat().st_size > 2:
    a = pd.read_csv(d / "aligned.csv")
    la, lb = int(tl["lap_a"]), int(tl["lap_b"])
    ra = laps[(laps.Driver == A) & (laps.LapNumber == la)].iloc[0]
    rb = laps[(laps.Driver == B) & (laps.LapNumber == lb)].iloc[0]
    want = rb.LapTimeS - ra.LapTimeS
    flag(abs(a.Delta.iloc[-1] - want) < 0.03, f"final delta {a.Delta.iloc[-1]:+.3f} s vs official lap-time difference {want:+.3f} s")
    flag(abs(a.T_A.iloc[-1] - ra.LapTimeS) < 0.05, f"telemetry duration {a.T_A.iloc[-1]:.3f} s vs official lap time {ra.LapTimeS:.3f} s (A)")
    flag(abs(tel["b_distance_scale"] - 1) < 0.01, f"distance rescale factor {tel['b_distance_scale']:.4f} (should be within 1%)")
    # delta at the sector lines: where A's clock reaches the S1 and S1+S2 splits
    for name, tA, want_s in [("S1", ra.S1S, rb.S1S - ra.S1S), ("S1+S2", ra.S1S + ra.S2S, (rb.S1S + rb.S2S) - (ra.S1S + ra.S2S))]:
        i = int(np.searchsorted(a.T_A.to_numpy(), tA))
        i = min(i, len(a) - 1)
        got = a.Delta.iloc[i]
        flag(abs(got - want_s) < 0.10, f"delta at {name} line ({a.Distance.iloc[i]:.0f} m): curve {got:+.3f} s vs official split difference {want_s:+.3f} s")
    # physically implausible jumps: delta change over 50 m as a share of the time A needs for those 50 m
    # (= the speed difference the curve implies; a 7 km/h gap at a 70 km/h hairpin is ~10 %, 15 %+ is suspect)
    step = 50.0
    dist = a.Distance.to_numpy(); dd = a.Delta.to_numpy(); ta = a.T_A.to_numpy()
    grid = np.arange(0, dist.max() - step, 5.0)
    jump = np.abs(np.interp(grid + step, dist, dd) - np.interp(grid, dist, dd))
    wt = np.maximum(np.interp(grid + step, dist, ta) - np.interp(grid, dist, ta), 1e-6)
    frac = jump / wt
    spd = a.Speed_A.to_numpy(float)
    s0, s1 = np.interp(grid, dist, spd), np.interp(grid + step, dist, spd)
    steady = np.abs(s1 - s0) / np.maximum((s0 + s1) / 2, 1.0) < 0.25   # skip braking/acceleration windows
    k = int(np.argmax(np.where(steady, frac, -1)))
    flag(frac[steady].max() < 0.15, f"largest delta change inside 50 m on steady-speed stretches: {jump[k]:.3f} s at ~{grid[k]:.0f} m = {frac[k]:.0%} implied speed difference")
    k2 = int(np.argmax(jump))
    print(f"      (largest jump overall: {jump[k2]:.3f} s at ~{grid[k2]:.0f} m; A speed {s0[k2]:.0f}->{s1[k2]:.0f} km/h; "
          f"steady window: {bool(steady[k2])}. In braking zones a 10-20 m position error alone gives big apparent differences.)")
else:
    print("      no telemetry in this bundle")