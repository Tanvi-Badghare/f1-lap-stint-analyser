"""OpenF1 loader tests against a SIMULATED OpenF1 server (documented schema, synthetic numbers). No internet."""
import json
import re
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd
import pytest

from f1analyser import openf1 as O
from f1analyser.config import AnalysisConfig
from f1analyser.demo import DRIVERS, make_laps

SK, T0 = 999, pd.Timestamp("2026-03-29T05:00:00Z")
CIRCUIT = {"corners": [{"number": 1, "letter": "", "trackPosition": {"x": 0.0, "y": 6000.0}},
                       {"number": 2, "letter": "", "trackPosition": {"x": 0.0, "y": -6000.0}}]}


class FakeOpenF1:
    """Serves OpenF1-shaped JSON for one 57-lap race. Lap date_start is jittered (documented as approximate)."""

    def __init__(self, fail_first=0):
        self.calls, self.fail_first = [], fail_first
        r = make_laps(n_laps=57, pit_laps=(24,), seed=3)
        r["start_s"] = (r["Time"] - r["LapTime"]).dt.total_seconds()
        r["dur_s"] = r["LapTime"].dt.total_seconds()
        self.raw, self.num = r, {d: i + 1 for i, d in enumerate(DRIVERS)}
        self.starts = {d: g.sort_values("LapNumber")["start_s"].to_numpy() for d, g in r.groupby("Driver")}
        self.durs = {d: g.sort_values("LapNumber")["dur_s"].to_numpy() for d, g in r.groupby("Driver")}
        a40 = self.starts["AAA"][39]
        a41 = self.starts["AAA"][40]
        iso = lambda s: (T0 + pd.Timedelta(seconds=float(s))).isoformat()
        self.rc = [
            dict(session_key=SK, date=iso(a40 + 30), category="SafetyCar", flag=None, scope=None, message="SAFETY CAR DEPLOYED", driver_number=None, lap_number=40),
            dict(session_key=SK, date=iso(a41 + 30), category="SafetyCar", flag=None, scope=None, message="SAFETY CAR IN THIS LAP", driver_number=None, lap_number=41),
            dict(session_key=SK, date=iso(1000), category="Other", flag=None, scope="Driver", message="CAR 2 (BBB) TIME 1:30.000 DELETED - TRACK LIMITS", driver_number=2, lap_number=12),
        ]
        self.meetings = [
            dict(year=2026, meeting_key=1, meeting_name="Australian Grand Prix", country_name="Australia", location="Melbourne", circuit_short_name="Melbourne", date_start="2026-03-08T00:00:00", is_cancelled=False),
            dict(year=2026, meeting_key=2, meeting_name="Chinese Grand Prix", country_name="China", location="Shanghai", circuit_short_name="Shanghai", date_start="2026-03-15T00:00:00", is_cancelled=False),
            dict(year=2026, meeting_key=3, meeting_name="Japanese Grand Prix", meeting_official_name="FORMULA 1 JAPANESE GRAND PRIX 2026", country_name="Japan", location="Suzuka", circuit_short_name="Suzuka", date_start="2026-03-29T00:00:00", is_cancelled=False, circuit_info_url="https://fake.example/circuit.json"),
            dict(year=2026, meeting_key=0, meeting_name="Pre-Season Testing", country_name="Bahrain", location="Sakhir", circuit_short_name="Sakhir", date_start="2026-02-01T00:00:00", is_cancelled=False),
        ]
        self.sessions = [dict(meeting_key=3, session_key=SK - 2, session_name="Practice 1", date_start="2026-03-27T02:00:00", is_cancelled=False),
                         dict(meeting_key=3, session_key=SK, session_name="Race", date_start="2026-03-29T05:00:00", is_cancelled=False)]

    # --- tables
    def laps(self):
        rows = []
        for r in self.raw.itertuples():
            jitter = 0.25 * np.sin(r.LapNumber * 1.7 + self.num[r.Driver])
            rows.append(dict(session_key=SK, driver_number=self.num[r.Driver], lap_number=int(r.LapNumber),
                             date_start=None if r.LapNumber == 1 else (T0 + pd.Timedelta(seconds=float(r.start_s + jitter))).isoformat(),
                             lap_duration=float(r.dur_s), duration_sector_1=r.Sector1Time.total_seconds(),
                             duration_sector_2=r.Sector2Time.total_seconds(), duration_sector_3=r.Sector3Time.total_seconds(),
                             is_pit_out_lap=bool(r.LapNumber == 25)))
        return rows

    def stints(self):
        out = []
        for d, n in self.num.items():
            out += [dict(session_key=SK, driver_number=n, stint_number=1, lap_start=1, lap_end=24, compound="MEDIUM", tyre_age_at_start=0),
                    dict(session_key=SK, driver_number=n, stint_number=2, lap_start=25, lap_end=57, compound="HARD", tyre_age_at_start=0)]
        return out

    def drivers(self):
        return [dict(session_key=SK, driver_number=n, name_acronym=d, team_name=f"Team{(n - 1) // 2}") for d, n in self.num.items()]

    def telemetry(self, endpoint, number, lo, hi):
        code = next(d for d, n in self.num.items() if n == number)
        phase, step = 0.05 * number, 0.27
        rows = []
        for k in range(int(np.ceil((lo - phase) / step)), int(np.floor((hi - phase) / step)) + 1):
            t = phase + k * step
            i = np.searchsorted(self.starts[code], t, side="right") - 1
            if i < 0 or t >= self.starts[code][i] + self.durs[code][i]:
                continue
            s = (t - self.starts[code][i]) / self.durs[code][i]
            date = (T0 + pd.Timedelta(seconds=float(t))).isoformat()
            if endpoint == "location":
                rows.append(dict(session_key=SK, driver_number=number, date=date, x=int(10000 * np.cos(2 * np.pi * s)), y=int(6000 * np.sin(2 * np.pi * s)), z=0))
            else:
                speed = int(220 + 80 * np.cos(4 * np.pi * s))
                sn = np.sin(4 * np.pi * s)
                rows.append(dict(session_key=SK, driver_number=number, date=date, speed=speed, throttle=100 if sn <= 0 else 0,
                                 brake=100 if (sn > 0.02 and speed < 260) else 0, n_gear=int(np.clip(speed // 40, 1, 8)), rpm=9000, drs=0))
        return rows

    # --- HTTP
    def http_get(self, url):
        self.calls.append(url)
        if self.fail_first > 0:
            self.fail_first -= 1
            return 429, "slow down"
        u = urlsplit(url)
        if u.netloc == "fake.example":
            return 200, json.dumps(CIRCUIT)
        endpoint = u.path.rsplit("/", 1)[-1]
        q = [re.match(r"^([a-z_]+)(>=|<=|>|<|=)(.*)$", unquote(p)).groups() for p in u.query.split("&") if p]
        eq = {k: v for k, op, v in q if op == "="}
        if endpoint == "meetings":
            data = [m for m in self.meetings if str(m["year"]) == eq.get("year", str(m["year"]))]
        elif endpoint == "sessions":
            data = [s for s in self.sessions if str(s["meeting_key"]) == eq["meeting_key"]]
        elif endpoint in ("car_data", "location"):
            lo = (pd.Timestamp(next(v for k, op, v in q if k == "date" and op == ">="), tz="UTC") - T0).total_seconds()
            hi = (pd.Timestamp(next(v for k, op, v in q if k == "date" and op == "<="), tz="UTC") - T0).total_seconds()
            data = self.telemetry(endpoint, int(eq["driver_number"]), lo, hi)
        else:
            data = {"laps": self.laps, "stints": self.stints, "drivers": self.drivers, "race_control": lambda: self.rc}[endpoint]()
        return (200, json.dumps(data)) if data else (404, '{"detail":"No results found."}')


def make_client(tmp_path, fake):
    return O.OpenF1Client(tmp_path / "cache", min_interval=0, http_get=fake.http_get, sleep=lambda s: None)


# ----------------------------------------------------------------------------- tests
def test_url_building():
    url = O.OpenF1Client.url("car_data", {"session_key": 9, "driver_number": 1}, ["date>=2026-03-29T05:00:00.000"])
    assert url == "https://api.openf1.org/v1/car_data?session_key=9&driver_number=1&date>=2026-03-29T05:00:00.000"
    assert "%20" in O.OpenF1Client.url("sessions", {"session_name": "Sprint Qualifying"})


def test_retry_cache_and_404(tmp_path):
    fake = FakeOpenF1(fail_first=2)
    c = make_client(tmp_path, fake)
    assert len(c.get("meetings", year=2026)) == 4 and c.network_calls == 3   # two 429s then success
    c.get("meetings", year=2026)
    assert c.network_calls == 3                                              # served from cache
    assert c.get("sessions", meeting_key=12345) == []                        # 404 -> empty list


def test_find_session(tmp_path):
    c = make_client(tmp_path, FakeOpenF1())
    assert O.find_session(c, 2026, "Japan", "R")["session_key"] == SK
    assert O.find_session(c, 2026, "3", "R")["event"] == "Japanese Grand Prix"   # round numbers skip testing
    with pytest.raises(O.OpenF1Error, match="No 2026 meeting"):
        O.find_session(c, 2026, "Atlantis", "R")
    with pytest.raises(O.OpenF1Error, match="no 'Qualifying'"):
        O.find_session(c, 2026, "Japan", "Q")


def test_build_laps_schema_and_derived_columns(tmp_path):
    fake = FakeOpenF1()
    laps, windows, teams = O.build_laps(fake.laps(), fake.stints(), fake.drivers(), fake.rc)
    a = laps[laps.Driver == "AAA"].set_index("LapNumber")
    assert a.loc[10, "Compound"] == "MEDIUM" and a.loc[30, "Compound"] == "HARD"
    assert a.loc[25, "TyreLife"] == 1 and a.loc[30, "TyreLife"] == 6 and a.loc[30, "Stint"] == 2
    assert pd.notna(a.loc[24, "PitInTime"]) and pd.notna(a.loc[25, "PitOutTime"]) and pd.isna(a.loc[30, "PitInTime"])
    assert a.loc[30, "TrackStatus"] == "1" and "4" in a.loc[40, "TrackStatus"] and "4" in a.loc[41, "TrackStatus"]
    assert bool(laps[(laps.Driver == "BBB") & (laps.LapNumber == 12)]["Deleted"].iat[0]) and not bool(a.loc[12, "Deleted"])
    assert pd.notna(a.loc[1, "Time"]) and a.loc[1, "LapTime"] > pd.Timedelta(seconds=80)  # lap 1 start filled from next lap
    assert a["IsAccurate"].all() and teams["AAA"] == teams["BBB"] == "Team0"
    assert windows[("AAA", 30)][2] == 1


def test_refine_start_finds_line_crossing():
    t0 = pd.Timestamp("2026-03-29T05:10:00Z")
    t = t0 + pd.to_timedelta(np.arange(-3, 3, 0.27), unit="s")
    true_cross = 0.4                                    # car crosses y = 0 at +0.4 s
    loc = pd.DataFrame({"t": t, "x": 10000, "y": ((t - t0).total_seconds() - true_cross) * 600.0})
    out = O.refine_start(loc, t0, (10000.0, 0.0), (0.0, 1.0))
    assert abs((out - t0).total_seconds() - true_cross) < 0.02
    assert O.refine_start(loc.iloc[:2], t0, (10000.0, 0.0), (0.0, 1.0)) == t0   # too little data -> unchanged


def test_corners_sanity_check():
    s = np.linspace(0, 1, 400, endpoint=False)
    tel = pd.DataFrame({"X": 10000 * np.cos(2 * np.pi * s), "Y": 6000 * np.sin(2 * np.pi * s), "Distance": s * 5500})
    c = O.corners_from_multiviewer(CIRCUIT, tel)
    assert list(c["Number"]) == [1, 2] and np.allclose(c["Distance"], [1375, 4125], atol=30)
    far = {"corners": [{"number": 1, "trackPosition": {"x": 5e6, "y": 5e6}}]}
    assert O.corners_from_multiviewer(far, tel) is None
    tenth = {"corners": [{"number": 1, "trackPosition": {"x": 0.0, "y": 60000.0}}]}   # 10x units are detected
    assert O.corners_from_multiviewer(tenth, tel) is not None


def test_end_to_end_openf1_analysis(tmp_path):
    fake = FakeOpenF1()
    cfg = AnalysisConfig(year=2026, gp="Japan", session="R", driver_a="AAA", driver_b="BBB", n_bootstrap=500,
                         cache_dir=str(tmp_path))
    res = O.compute_openf1(cfg, client=make_client(tmp_path, fake))
    p, fr = res["pace"], res["frames"]
    assert p["paired_n"] > 30 and p["paired_ci_low"] < 0.30 < p["paired_ci_high"]
    al, info = fr["aligned"], res["tel_info"]
    assert len(al) > 800 and abs(info["b_distance_scale"] - 1) < 0.01
    tl = res["tel_laps"]
    assert abs(info["lap_time_a_s"] - fake.durs["AAA"][tl["lap_a"] - 1]) < 1e-6       # telemetry spans the official lap time
    assert abs(info["lap_time_b_s"] - fake.durs["BBB"][tl["lap_b"] - 1]) < 1e-6
    assert abs(al["Delta"].iloc[-1] - (info["lap_time_b_s"] - info["lap_time_a_s"])) < 0.01   # 5 m grid stops just short
    assert abs(al["Delta"].iloc[len(al) // 2] - al["Delta"].iloc[-1] / 2) < 0.06      # smooth ramp, no start-line offset
    assert len(fr["corners"]) == 2 and fr["corners"]["MinSpeed_AAA_kmh"].between(135, 150).all()
    assert {"pace_gap", "stint_trend", "limits"} <= {f["id"] for f in res["findings"]}
    assert res["meta"]["teams"] == {"AAA": "Team0", "BBB": "Team0"} and "OpenF1" in res["meta"]["data_source"]
    assert len(fake.calls) <= 14                                                   # polite: a handful of requests


def test_screen_openf1(tmp_path):
    pairs = O.screen_openf1(2026, "Japan", "R", str(tmp_path), client=make_client(tmp_path, FakeOpenF1()))
    assert {"driver_a", "driver_b", "min_clear_air"} <= set(pairs.columns) and len(pairs) >= 3
    assert set(pairs.iloc[0][["driver_a", "driver_b"]]) <= set(DRIVERS)


def test_website_live_flow_uses_openf1(tmp_path, monkeypatch):
    """Click through the website: Run live analysis -> OpenF1 -> drivers -> Run analysis -> metrics render."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    fake = FakeOpenF1()
    monkeypatch.setattr(O.OpenF1Client, "_requests_get", staticmethod(fake.http_get))
    monkeypatch.setenv("F1_WEB_DATA", str(tmp_path / "web_data"))
    app = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=180).run()
    at.sidebar.radio[0].set_value("Run live analysis").run()
    assert not at.exception, at.exception
    assert at.sidebar.selectbox[0].value.startswith("OpenF1")
    by_label = lambda widgets, text: next(w for w in widgets if w.label.startswith(text))
    by_label(at.sidebar.text_input, "Cache").set_value(str(tmp_path))
    by_label(at.sidebar.text_input, "Driver A").set_value("AAA")
    by_label(at.sidebar.text_input, "Driver B").set_value("BBB").run()
    by_label(at.sidebar.button, "2. Run analysis").click().run(timeout=180)
    assert not at.exception, at.exception
    assert not at.sidebar.error, [e.value for e in at.sidebar.error]
    assert len(at.metric) == 5 and at.session_state["live"]["meta"]["openf1_session_key"] == SK
    by_label(at.sidebar.button, "Save to website folder").click().run(timeout=120)
    assert (tmp_path / "web_data").exists() and any((tmp_path / "web_data").glob("*/bundle.json"))


def test_pick_pair():
    import pandas as pd
    from f1analyser.build_season import _pick_pair
    ok = pd.DataFrame([dict(team="T", driver_a="AAA", driver_b="BBB", min_clean=20, min_clear_air=10)])
    assert _pick_pair(ok) == ("AAA", "BBB")
    assert _pick_pair(ok.assign(min_clean=3)) is None
    assert _pick_pair(pd.DataFrame()) is None


def test_analysis_survives_telemetry_401(tmp_path):
    from f1analyser.service import save_bundle, load_bundle

    class Deny(FakeOpenF1):
        def http_get(self, url):
            if "/car_data" in url or "/location" in url:
                return 401, "unauthorised"
            return super().http_get(url)

    fake = Deny()
    cfg = AnalysisConfig(year=2026, gp="Japan", session="R", driver_a="AAA", driver_b="BBB", n_bootstrap=300,
                         cache_dir=str(tmp_path))
    res = O.compute_openf1(cfg, client=make_client(tmp_path, fake))
    assert res["tel_laps"] == {} and res["tel_info"] == {} and res["frames"]["aligned"].empty
    assert res["pace"]["paired_n"] > 30 and "telemetry_note" in res["meta"]
    limits = next(f for f in res["findings"] if f["id"] == "limits")
    assert "Telemetry could not be loaded" in limits["evidence"]
    assert all(f["id"] != "telemetry_localisation" for f in res["findings"])
    back = load_bundle(save_bundle(res, tmp_path / "web"))
    assert back["frames"]["aligned"].empty and back["tel_laps"] == {}


def test_no_telemetry_flag(tmp_path):
    fake = FakeOpenF1()
    cfg = AnalysisConfig(year=2026, gp="Japan", session="R", driver_a="AAA", driver_b="BBB", n_bootstrap=300,
                         cache_dir=str(tmp_path), use_telemetry=False)
    res = O.compute_openf1(cfg, client=make_client(tmp_path, fake))
    assert res["tel_laps"] == {} and not any("/car_data" in c or "/location" in c for c in fake.calls)


def test_telemetry_401_falls_back_to_lap_analysis():
    """A 401 on location/car_data must not kill the lap, sector and stint analysis."""
    from f1analyser.config import AnalysisConfig
    from f1analyser.demo import make_laps
    from f1analyser.openf1 import OpenF1Error
    from f1analyser.service import compute_all
    import types
    laps = make_laps()
    drv = sorted(laps["Driver"].unique())[:2]
    sess = types.SimpleNamespace(laps=laps)

    def boom(*a, **k):
        raise OpenF1Error("OpenF1 refused the request (HTTP 401)")
    cfg = AnalysisConfig(year=2026, gp="X", session="R", driver_a=drv[0], driver_b=drv[1])
    res = compute_all(cfg, session=sess, tel_fn=boom, meta=dict(event="X", teams={}, session_name="Race", session_date="", data_source="t"))
    assert len(res["frames"]["al"] if "al" in res["frames"] else res["frames"]["aligned"]) == 0
    assert "Telemetry could not be loaded" in " ".join(f["evidence"] for f in res["findings"])


def test_build_web_no_telemetry_flag():
    import argparse, inspect
    from f1analyser import build_web
    assert "--no-telemetry" in inspect.getsource(build_web)


def test_stale_speed_run_in_braking_zone_is_dropped():
    rows = []
    sp = [250, 240, 230, 210, 190, 170, 150, 138, 138, 138, 138, 125, 108, 95, 100]
    for k, v in enumerate(sp):
        rows.append(dict(date=(T0 + pd.Timedelta(seconds=0.27 * k)).isoformat(), speed=v, brake=100, throttle=0, n_gear=4, rpm=9000))
    cd, n = O.drop_stale_braking_samples(pd.DataFrame(rows))
    assert n == 4 and (cd["speed"] == 138).sum() == 0 and len(cd) == len(sp) - 4
    # steady speed with the brake off (a normal straight) must be left alone
    straight = pd.DataFrame([dict(date=(T0 + pd.Timedelta(seconds=0.27 * k)).isoformat(), speed=274, brake=0, throttle=100, n_gear=8, rpm=9000) for k in range(8)])
    assert O.drop_stale_braking_samples(straight)[1] == 0
