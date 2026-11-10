"""Website layer: compute -> bundle -> plots -> Streamlit app, all on SYNTHETIC data (no internet)."""
from pathlib import Path

import numpy as np
import pytest

from f1analyser import webplots as W
from f1analyser.demo import demo_result
from f1analyser.service import list_bundles, load_bundle, save_bundle

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def demo():
    return demo_result()


def test_compute_recovers_known_values(demo):
    p = demo["pace"]
    assert p["paired_ci_low"] < 0.30 < p["paired_ci_high"]
    al = demo["frames"]["aligned"]
    assert abs(al["Delta"].iloc[-1] - 0.30) < 0.02
    worst = demo["frames"]["minis"].sort_values("A_gain_s").iloc[-1]
    assert 1800 <= worst["StartM"] <= 2500
    assert {"pace_gap", "stint_trend", "limits"} <= {f["id"] for f in demo["findings"]}


def test_bundle_round_trip(demo, tmp_path):
    out = save_bundle(demo, tmp_path)
    assert list_bundles(tmp_path) == [out]
    r = load_bundle(out)
    for name in ("laps", "aligned", "minis", "stint_table", "paired"):
        assert len(r["frames"][name]) == len(demo["frames"][name])
    assert r["frames"]["laps"]["Clean"].dtype == bool
    assert r["frames"]["aligned"]["Brake_A"].dtype == bool
    assert np.isclose(r["pace"]["paired_median"], demo["pace"]["paired_median"])


def test_all_figures_build(demo, tmp_path):
    r = load_bundle(save_bundle(demo, tmp_path))
    fr, a, b = r["frames"], "AAA", "BBB"
    corners = fr["corners"] if len(fr["corners"]) else None
    figs = [
        W.lap_times_fig(fr["laps"], a, b), W.paired_delta_fig(fr["paired"], r["pace"], a, b),
        W.sector_fig(fr["sectors"], a, b), W.telemetry_fig(fr["aligned"], a, b, corners),
        W.minisector_fig(fr["minis"], a, b), W.dominance_fig(fr["aligned"], fr["minis"], a, b),
        W.stints_fig(fr["laps"], fr["stint_table"], [a, b]), W.relpace_fig(fr["laps"], a, b),
        W.fuel_fig(fr["fuel_sensitivity"]),
    ]
    assert all(f is not None and len(f.data) > 0 for f in figs)


def test_streamlit_app_demo_and_saved(demo, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    # saved-analysis mode
    save_bundle(demo, tmp_path)
    monkeypatch.setenv("F1_WEB_DATA", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.sidebar.radio[0].value == "Saved analyses"
    assert any("SYNTHETIC" in w.value for w in at.warning)
    assert len(at.tabs) == 7 and len(at.metric) == 5

    # demo mode with an empty data folder
    monkeypatch.setenv("F1_WEB_DATA", str(tmp_path / "empty"))
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.sidebar.radio[0].value == "Run live analysis" or "Demo (synthetic data)" in at.sidebar.radio[0].options
    at.sidebar.radio[0].set_value("Demo (synthetic data)").run()
    assert not at.exception, at.exception
    assert len(at.metric) == 5
