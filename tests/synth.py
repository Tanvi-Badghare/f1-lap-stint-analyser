"""Synthetic data generators that mimic the FastF1 Laps / telemetry schema."""
import numpy as np
import pandas as pd

DRIVERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]


def make_laps(n_laps=50, pit_laps=(25,), seed=1, gap_b=0.30, deg=0.05, fuel=0.03):
    """AAA is faster than BBB by gap_b s. True slope within stint = deg - fuel (s/lap)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, drv in enumerate(DRIVERS):
        k_eff = 0 if drv in ("AAA", "BBB") else k  # AAA/BBB differ only by gap_b
        base = 90.0 + 0.15 * k_eff + (gap_b if drv == "BBB" else 0.0)
        stint, age, comp, t_end = 1, 1, "MEDIUM", 0.0
        for lap in range(1, n_laps + 1):
            pit_in = pit_out = pd.NaT
            if lap in pit_laps:
                pit_in = pd.Timedelta(seconds=t_end)
            if lap - 1 in pit_laps:
                stint, age, comp = stint + 1, 1, "HARD"
                pit_out = pd.Timedelta(seconds=t_end)
            lt = base + deg * age - fuel * lap + rng.normal(0, 0.12)
            if lap in pit_laps or lap - 1 in pit_laps:
                lt += 20
            t_end += lt
            rows.append(dict(
                Driver=drv, LapNumber=lap, LapTime=pd.Timedelta(seconds=lt),
                Sector1Time=pd.Timedelta(seconds=lt * 0.30 + (0.05 if drv == "BBB" else 0)),
                Sector2Time=pd.Timedelta(seconds=lt * 0.40 + (0.20 if drv == "BBB" else 0)),
                Sector3Time=pd.Timedelta(seconds=lt * 0.30 + (0.05 if drv == "BBB" else 0)),
                Stint=stint, Compound=comp, TyreLife=float(age), TrackStatus="1", IsAccurate=True,
                Deleted=False, PitInTime=pit_in, PitOutTime=pit_out, Time=pd.Timedelta(seconds=t_end)))
            age += 1
    return pd.DataFrame(rows)


def make_telemetry(lap_time=90.0, length=5000.0, hz=5, b_slower_zone=None, seed=0):
    """One lap: speed profile with two slow corners. b_slower_zone=(d0, d1, extra_seconds) adds time loss."""
    n = int(lap_time * hz)
    s = np.linspace(0, 1, n)
    dist = s * length
    speed = 220 + 80 * np.cos(2 * np.pi * 2 * s)  # two slow zones (min at s=0.5, ~140)
    brake = (np.gradient(speed) < -0.3) & (speed < 260)
    throttle = np.where(np.gradient(speed) >= 0, 100.0, 0.0)
    t = np.linspace(0, lap_time, n)
    if b_slower_zone:
        d0, d1, extra = b_slower_zone
        w = ((dist - d0) / (d1 - d0)).clip(0, 1)
        t = t + extra * w          # time loss accrued inside the zone
    return pd.DataFrame({
        "Distance": dist, "T": t, "Time": pd.to_timedelta(t, unit="s"), "Speed": speed,
        "Throttle": throttle, "Brake": brake, "nGear": np.clip((speed // 40).astype(int), 1, 8),
        "X": 1000 * np.cos(2 * np.pi * s), "Y": 600 * np.sin(2 * np.pi * s)})
