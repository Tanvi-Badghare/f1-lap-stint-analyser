import numpy as np
import pandas as pd

from f1analyser import telemetry as T
from f1analyser import plots as P
from f1analyser.config import AnalysisConfig
from f1analyser.findings import build_findings
from f1analyser.pipeline import analyse_laps
from f1analyser.report import render_report
from synth import make_laps, make_telemetry


def _aligned(extra=0.30):
    ta = make_telemetry(90.0)
    tb = make_telemetry(90.0 + extra, b_slower_zone=(2000, 2500, 0.0))
    tb = make_telemetry(90.0, b_slower_zone=(2000, 2500, extra))   # B loses `extra` s inside 2000-2500 m
    return T.align_laps(ta, tb, 5.0)


def test_delta_end_equals_lap_time_difference():
    al, info = _aligned(0.30)
    assert abs(al["Delta"].iloc[-1] - 0.30) < 0.02


def test_minisector_localises_time_loss():
    al, _ = _aligned(0.30)
    m = T.minisector_deltas(al, 25)
    worst = m.loc[m["A_gain_s"].idxmax()]
    assert 1800 <= worst["StartM"] <= 2500
    assert abs(m["A_gain_s"].sum() - 0.30) < 0.02
    assert m.loc[m["EndM"] < 1900, "A_gain_s"].abs().max() < 0.02


def test_corner_units_defensive():
    c = pd.DataFrame({"Number": [1], "Letter": [""], "Distance": [25000.0]})
    assert T.normalise_corners(c, 5000.0)["Distance"].iat[0] == 2500.0


def test_corner_table_columns():
    al, info = _aligned(0.30)
    corners = pd.DataFrame({"Number": [1, 2], "Letter": ["", ""], "Distance": [1250.0, 3750.0]})
    ct = T.corner_table(al, corners, "AAA", "BBB")
    assert {"EntryGain_A_s", "ExitGain_A_s", "A_brakes_later_m", "MinSpeed_AAA_kmh"} <= set(ct.columns)
    assert len(ct) == 2


def test_end_to_end_findings_plots_report(tmp_path):
    cfg = AnalysisConfig(year=2000, gp="Test", driver_a="AAA", driver_b="BBB", n_bootstrap=500)
    res = analyse_laps(make_laps(), cfg)
    al, info = _aligned(0.30)
    minis = T.minisector_deltas(al, 25)
    corners = pd.DataFrame({"Number": [1, 2], "Letter": ["", ""], "Distance": [1250.0, 3750.0]})
    ct = T.corner_table(al, corners, "AAA", "BBB")
    tl = dict(lap_a=30, lap_b=30, mode="matched", reason="test")
    f = build_findings("AAA", "BBB", res.pace, res.tables["sectors"], res.tables["stint_table"],
                       res.tables["fuel_sensitivity"], minis, ct, info, tl)
    ids = [x["id"] for x in f]
    assert {"pace_gap", "sector_location", "telemetry_localisation", "stint_trend", "limits"} <= set(ids)
    # guard-rail: no finding may claim to have isolated degradation or a setup cause
    text = " ".join(x["title"] + x["evidence"] for x in f).lower()
    assert "caused by" not in text and "setup problem" not in text
    figs = {
        "telemetry": P.plot_telemetry(al, "AAA", "BBB", corners, tmp_path / "t.png"),
        "dom": P.plot_dominance_map(al, minis, "AAA", "BBB", tmp_path / "d.png"),
        "bars": P.plot_minisector_bars(minis, "AAA", "BBB", tmp_path / "b.png"),
        "stints": P.plot_stints(res.clean, res.all_laps, ["AAA", "BBB"], "AAA", res.tables["stint_table"], tmp_path / "s.png"),
        "rel": P.plot_relative_pace(res.clean, ["AAA", "BBB"], "AAA", tmp_path / "r.png"),
        "fuel": P.plot_fuel_sensitivity(res.tables["fuel_sensitivity"], tmp_path / "f.png"),
        "pair": P.plot_paired_delta(res.paired, res.pace, "AAA", "BBB", tmp_path / "p.png"),
        "dist": P.plot_lap_distribution(res.clean[res.clean.Driver.isin(["AAA", "BBB"])], "AAA", "BBB", tmp_path / "l.png"),
        "sec": P.plot_sector_deltas(res.tables["sectors"], "AAA", "BBB", tmp_path / "x.png"),
    }
    assert all((tmp_path / p.split("/")[-1]).exists() for p in figs.values())
    tables = dict(res.tables, minisectors=minis, corners=ct, channel_summary=T.channel_summary(al, "AAA", "BBB"))
    md = render_report(cfg, {"event": "Test GP", "command": "x"}, tables, figs, f, tl, info)
    assert "Your interpretation" in md and "## 6. Observations" in md


def test_select_laps_matched_falls_back_visibly():
    cfg = AnalysisConfig(year=2000, gp="Test", driver_a="AAA", driver_b="BBB")
    res = analyse_laps(make_laps(), cfg)
    sel = T.select_comparison_laps(res.clean[res.clean.Driver.isin(["AAA", "BBB"])], "AAA", "BBB", "matched")
    assert sel["mode"] in ("matched", "fastest") and sel["reason"]
    assert T.select_comparison_laps(res.clean, "AAA", "BBB", explicit=(31, 31))["mode"] == "explicit"


def test_path_alignment_recovers_known_gap_despite_speed_bias_and_position_noise():
    """Figure-eight track. B is 7 % slower in 2000-2500 m (a known time loss). B's speed sensor has a 1 % slow bias
    drift (so integrated distance drifts) and both cars' X/Y carry noise. Truth is known exactly from the model."""
    import numpy as np
    import pandas as pd
    from f1analyser.telemetry import align_laps

    rng = np.random.default_rng(3)
    L = 5800.0
    sg = np.linspace(0, L, 58001)

    def vprof(slow):
        v = 62 + 18 * np.cos(2 * np.pi * 3 * sg / L) + 8 * np.cos(2 * np.pi * 11 * sg / L)
        return np.where((sg > 2000) & (sg < 2500), v * 0.93, v) if slow else v

    def truth(slow):
        v = vprof(slow)
        return np.r_[0, np.cumsum(np.diff(sg) / (0.5 * (v[1:] + v[:-1])))]

    def sample(slow, bias, noise, hz=3.7):
        tt = truth(slow)
        t = np.arange(0, tt[-1], 1 / hz)
        s = np.interp(t, tt, sg)
        v = np.interp(s, sg, vprof(slow))
        vm = np.round(v * 3.6 * (1 + bias * np.sin(2 * np.pi * t / tt[-1] + 0.7))) / 3.6
        dist = np.r_[0, np.cumsum(0.5 * (vm[1:] + vm[:-1]) * np.diff(t))]
        th = 2 * np.pi * s / L
        return pd.DataFrame(dict(Distance=dist, T=t, Speed=vm * 3.6, Throttle=100.0, Brake=False, nGear=7,
                                 X=3000 * np.sin(th) + rng.normal(0, noise, len(t)),
                                 Y=1500 * np.sin(2 * th) + rng.normal(0, noise, len(t))))

    al, info = align_laps(sample(False, 0.0, 15), sample(True, 0.01, 15))
    true = np.interp(al["Distance"], sg, truth(True)) - np.interp(al["Distance"], sg, truth(False))
    assert info["alignment"].startswith("track position")
    assert np.abs(al["Delta"].to_numpy() - true).max() < 0.08
    assert info["max_implied_speed_diff_frac"] < 0.15
