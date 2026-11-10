import numpy as np
import pandas as pd

from f1analyser.config import AnalysisConfig
from f1analyser.pipeline import analyse_laps
from synth import make_laps

CFG = AnalysisConfig(year=2000, gp="Test", driver_a="AAA", driver_b="BBB", n_bootstrap=1000)


def test_cleaning_excludes_expected_laps():
    raw = make_laps()
    raw.loc[(raw.Driver == "AAA") & (raw.LapNumber == 10), "TrackStatus"] = "4"
    raw.loc[(raw.Driver == "AAA") & (raw.LapNumber == 11), "Deleted"] = True
    res = analyse_laps(raw, CFG)
    a = res.all_laps[res.all_laps.Driver == "AAA"].set_index("LapNumber")
    assert a.loc[1, "ExclusionReason"] == "lap_1"
    assert a.loc[10, "ExclusionReason"] == "not_green"
    assert a.loc[11, "ExclusionReason"] == "deleted_track_limits"
    assert a.loc[25, "ExclusionReason"] == "in_or_out_lap"
    assert a.loc[26, "ExclusionReason"] == "in_or_out_lap"
    assert a.loc[30, "Clean"]


def test_pace_gap_recovered_with_ci():
    res = analyse_laps(make_laps(gap_b=0.30), CFG)
    p = res.pace
    assert p["paired_n"] > 30
    assert p["paired_ci_low"] < 0.30 < p["paired_ci_high"]
    assert p["share_laps_A_faster"] > 0.9


def test_sector_split_matches_construction():
    res = analyse_laps(make_laps(gap_b=0.30), CFG)
    s = res.tables["sectors"].set_index("Sector")
    # S2 was built to carry the largest part of the gap
    assert s["median_delta_B_minus_A"].idxmax() == "S2"


def test_stint_slope_is_apparent_trend_and_fuel_assumption_shifts_it():
    res = analyse_laps(make_laps(deg=0.05, fuel=0.03), CFG)
    st = res.tables["stint_table"]
    assert (st["slope_raw"].sub(0.02).abs() < 0.02).all()          # deg - fuel = +0.02
    f = res.tables["fuel_sensitivity"]
    s0 = f[(f.Driver == "AAA") & (f.Stint == 1) & (f.assumed_fuel_effect_s_per_lap == 0.0)]["slope"].iat[0]
    s3 = f[(f.Driver == "AAA") & (f.Stint == 1) & (np.isclose(f.assumed_fuel_effect_s_per_lap, 0.03))]["slope"].iat[0]
    assert abs((s3 - s0) - 0.03) < 1e-6                              # fuel assumption adds exactly f per lap


def test_traffic_flag():
    raw = make_laps()
    # make BBB finish lap 30 0.5 s behind AAA
    ta = raw[(raw.Driver == "AAA") & (raw.LapNumber == 30)]["Time"].iat[0]
    raw.loc[(raw.Driver == "BBB") & (raw.LapNumber == 30), "Time"] = ta + pd.Timedelta(seconds=0.5)
    res = analyse_laps(raw, CFG)
    b30 = res.all_laps[(res.all_laps.Driver == "BBB") & (res.all_laps.LapNumber == 30)]
    assert bool(b30["InTraffic"].iat[0])
