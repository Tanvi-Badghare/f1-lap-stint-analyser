"""OpenF1 (https://openf1.org) data source.

Turns OpenF1 JSON into the same tables the FastF1 path produces, so everything downstream (cleaning, pace gap,
telemetry alignment, stints, findings, website) works unchanged.

Built against OpenF1's documented schema (laps, stints, drivers, race_control, car_data, location, meetings, sessions).
Known differences from FastF1 data (also listed in docs/DATA_SOURCE_OPENF1.md):
  * lap ``date_start`` is documented as approximate -> telemetry start is re-anchored on track position (refine_start)
  * no TrackStatus / IsAccurate / Deleted columns -> derived from race_control messages and lap completeness
  * no corner annotations -> taken from the MultiViewer circuit file linked in ``meetings`` (best effort)
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np
import pandas as pd

BASE = "https://api.openf1.org/v1"
SESSION_NAMES = {"R": ["Race"], "Q": ["Qualifying"], "S": ["Sprint"], "SQ": ["Sprint Qualifying", "Sprint Shootout"],
                 "FP1": ["Practice 1"], "FP2": ["Practice 2"], "FP3": ["Practice 3"]}


class OpenF1Error(RuntimeError):
    pass


# ----------------------------------------------------------------------------- HTTP client
class OpenF1Client:
    """Tiny GET client: on-disk cache, request spacing, retry with back-off on 429/5xx."""

    def __init__(self, cache_dir=".openf1_cache", min_interval=0.5, retries=6, http_get=None, sleep=time.sleep):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.min_interval, self.retries, self._sleep = min_interval, retries, sleep
        self._http_get = http_get or self._requests_get
        self._token, self._token_at = None, 0.0
        self._last = 0.0
        self.network_calls = 0

    def _auth_header(self) -> dict:
        """Optional OpenF1 login: set OPENF1_USERNAME and OPENF1_PASSWORD (or OPENF1_TOKEN). Tokens last 1 hour."""
        import os
        import requests
        tok = os.environ.get("OPENF1_TOKEN")
        user, pw = os.environ.get("OPENF1_USERNAME"), os.environ.get("OPENF1_PASSWORD")
        if not tok and user and pw:
            if self._token is None or time.monotonic() - self._token_at > 3000:
                r = requests.post(f"{BASE.rsplit('/v1', 1)[0]}/token", data={"username": user, "password": pw}, timeout=60)
                if r.status_code != 200:
                    raise OpenF1Error(f"OpenF1 login failed (HTTP {r.status_code}). Check OPENF1_USERNAME / OPENF1_PASSWORD.")
                self._token, self._token_at = r.json()["access_token"], time.monotonic()
            tok = self._token
        return {"Authorization": f"Bearer {tok}"} if tok else {}

    def _requests_get(self, url: str):
        import requests
        r = requests.get(url, timeout=90, headers={"User-Agent": "f1analyser/0.1", **self._auth_header()})
        return r.status_code, r.text

    @staticmethod
    def url(endpoint: str, params: dict | None = None, filters: list[str] | None = None) -> str:
        parts = [f"{k}={quote(str(v), safe=':-.T')}" for k, v in (params or {}).items()]
        parts += list(filters or [])  # raw "date>=2026-03-29T05:00:00.000" style filters
        return f"{BASE}/{endpoint}" + ("?" + "&".join(parts) if parts else "")

    def get(self, endpoint: str, filters: list[str] | None = None, **params) -> list[dict]:
        url = self.url(endpoint, params, filters)
        path = self.cache / f"{endpoint}_{hashlib.sha1(url.encode()).hexdigest()[:20]}.json"
        if path.exists():
            return json.loads(path.read_text())
        for attempt in range(self.retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = time.monotonic()
            self.network_calls += 1
            status, text = self._http_get(url)
            if status == 200:
                data = json.loads(text)
                path.write_text(json.dumps(data))
                return data
            if status == 404:
                return []  # OpenF1 answers 404 "No results found" for an empty query
            if status in (429, 500, 502, 503, 504):
                self._sleep(min(2 ** attempt, 30))
                continue
            if status in (401, 403):
                raise OpenF1Error(f"OpenF1 refused the request (HTTP {status}) for {url}. Recent sessions may need "
                                  f"authentication. OpenF1 now requires a (sponsor) account for some data such as location/car_data; "
                                  f"set OPENF1_USERNAME and OPENF1_PASSWORD to use one.")
            raise OpenF1Error(f"OpenF1 error HTTP {status} for {url}: {text[:200]}")
        raise OpenF1Error(f"OpenF1 kept answering with errors/rate limits for {url}. Wait a minute and retry; "
                          f"cached results are kept, so a retry continues where it stopped.")

    def fetch_json(self, url: str):
        """Fetch an arbitrary JSON URL (used for the MultiViewer circuit file)."""
        path = self.cache / f"ext_{hashlib.sha1(url.encode()).hexdigest()[:20]}.json"
        if path.exists():
            return json.loads(path.read_text())
        status, text = self._http_get(url)
        if status != 200:
            raise OpenF1Error(f"HTTP {status} for {url}")
        path.write_text(text)
        return json.loads(text)


def _ts(t: pd.Timestamp) -> str:
    """Naive-UTC ISO string for OpenF1 filters (avoids '+' in URLs)."""
    return t.tz_convert("UTC").tz_localize(None).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


# ----------------------------------------------------------------------------- session lookup
def find_session(client: OpenF1Client, year: int, gp, session_code: str = "R") -> dict:
    meetings = client.get("meetings", year=int(year))
    race_meetings = [m for m in meetings if not m.get("is_cancelled") and "test" not in str(m.get("meeting_name", "")).lower()]
    race_meetings.sort(key=lambda m: m.get("date_start", ""))
    if not race_meetings:
        raise OpenF1Error(f"OpenF1 returned no meetings for {year}. Check the year, or try again later.")
    g = str(gp).strip().lower()
    if g.isdigit():
        n = int(g)
        if not 1 <= n <= len(race_meetings):
            raise OpenF1Error(f"Round {n} not found; OpenF1 lists {len(race_meetings)} race weekends for {year}.")
        meeting = race_meetings[n - 1]
    else:
        keys = ("meeting_name", "meeting_official_name", "country_name", "location", "circuit_short_name")
        hits = [m for m in race_meetings if any(g in str(m.get(k, "")).lower() for k in keys)]
        if not hits:
            names = ", ".join(str(m.get("meeting_name")) for m in race_meetings)
            raise OpenF1Error(f"No {year} meeting matches '{gp}'. Available: {names}")
        if len(hits) > 1:
            names = ", ".join(str(m.get("meeting_name")) for m in hits)
            raise OpenF1Error(f"'{gp}' matches several meetings ({names}); be more specific or use the round number.")
        meeting = hits[0]
    wanted = SESSION_NAMES.get(str(session_code).upper(), [str(session_code)])
    sessions = client.get("sessions", meeting_key=meeting["meeting_key"])
    for name in wanted:
        for s in sessions:
            if str(s.get("session_name", "")).lower() == name.lower() and not s.get("is_cancelled"):
                return dict(session_key=s["session_key"], meeting_key=meeting["meeting_key"],
                            event=meeting.get("meeting_name"), official=meeting.get("meeting_official_name"),
                            location=meeting.get("location"), country=meeting.get("country_name"),
                            session_name=s.get("session_name"), date_start=s.get("date_start"),
                            circuit_info_url=meeting.get("circuit_info_url"))
    have = ", ".join(str(s.get("session_name")) for s in sessions)
    raise OpenF1Error(f"{meeting.get('meeting_name')} has no '{wanted[0]}' session in OpenF1 (has: {have}).")


# ----------------------------------------------------------------------------- laps
def status_windows(rc_rows: list[dict]) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Safety car / VSC / red flag / track-wide yellow windows from race-control messages.
    Codes follow FastF1: 2 yellow, 4 SC, 5 red, 6 VSC. Sector-only yellows are NOT included."""
    if not rc_rows:
        return []
    rc = pd.DataFrame(rc_rows)
    rc["date"] = pd.to_datetime(rc["date"], utc=True, errors="coerce")
    rc = rc.dropna(subset=["date"]).sort_values("date")
    wins, o = [], dict(sc=None, vsc=None, red=None, yel=None)
    for r in rc.itertuples():
        cat = str(getattr(r, "category", "") or "").upper()
        flag = str(getattr(r, "flag", "") or "").upper()
        msg = str(getattr(r, "message", "") or "").upper()
        scope = str(getattr(r, "scope", "") or "").upper()
        t = r.date
        if cat == "SAFETYCAR":
            if "VIRTUAL" in msg:
                if "DEPLOYED" in msg:
                    o["vsc"] = o["vsc"] or t
                elif "ENDING" in msg and o["vsc"] is not None:
                    wins.append(("6", o["vsc"], t)); o["vsc"] = None
            else:
                if "DEPLOYED" in msg:
                    o["sc"] = o["sc"] or t
                elif "IN THIS LAP" in msg and o["sc"] is not None:
                    wins.append(("4", o["sc"], t)); o["sc"] = None
        if flag == "RED" and scope in ("TRACK", ""):
            o["red"] = o["red"] or t
        elif flag in ("GREEN", "CLEAR") and scope == "TRACK":
            if o["red"] is not None:
                wins.append(("5", o["red"], t)); o["red"] = None
            if o["yel"] is not None:
                wins.append(("2", o["yel"], t)); o["yel"] = None
        if flag in ("YELLOW", "DOUBLE YELLOW") and scope == "TRACK":
            o["yel"] = o["yel"] or t
    end = rc["date"].max() + pd.Timedelta(hours=1)
    for key, code in (("sc", "4"), ("vsc", "6"), ("red", "5"), ("yel", "2")):
        if o[key] is not None:
            wins.append((code, o[key], end))
    return wins


def build_laps(laps_rows, stints_rows, drivers_rows, rc_rows):
    """OpenF1 rows -> FastF1-style Laps DataFrame, plus per-lap windows for telemetry fetching."""
    if not laps_rows:
        raise OpenF1Error("OpenF1 returned no laps for this session.")
    drv = {int(d["driver_number"]): d for d in drivers_rows}
    L = pd.DataFrame(laps_rows)
    for c in ("date_start", "lap_duration", "duration_sector_1", "duration_sector_2", "duration_sector_3", "is_pit_out_lap"):
        if c not in L:
            L[c] = np.nan
    L["driver_number"] = L["driver_number"].astype(int)
    L["Driver"] = L["driver_number"].map(lambda n: (drv.get(n) or {}).get("name_acronym") or str(n))
    L["start"] = pd.to_datetime(L["date_start"], utc=True, errors="coerce")
    L["dur"] = pd.to_numeric(L["lap_duration"], errors="coerce")
    L = L.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)
    nxt = L.groupby("driver_number")["start"].shift(-1)
    L["end"] = nxt.where(nxt.notna(), L["start"] + pd.to_timedelta(L["dur"], unit="s"))
    L["start"] = L["start"].fillna(L["end"] - pd.to_timedelta(L["dur"], unit="s"))
    t0 = L["start"].min()
    L["TimeS"] = (L["end"] - t0).dt.total_seconds()

    # stints -> compound, stint number, tyre age
    L["Stint"], L["Compound"], L["TyreLife"] = np.nan, "UNKNOWN", np.nan
    st = pd.DataFrame(stints_rows) if stints_rows else pd.DataFrame()
    pit_in, pit_out = set(), set()
    if len(st):
        for c in ("lap_start", "lap_end", "compound", "tyre_age_at_start", "stint_number"):
            if c not in st:
                st[c] = np.nan
        st["driver_number"] = st["driver_number"].astype(int)
        for d, g in st.sort_values("stint_number").groupby("driver_number"):
            rows = g.to_dict("records")
            for i, r in enumerate(rows):
                ls = 1 if pd.isna(r["lap_start"]) else int(r["lap_start"])
                le = 10 ** 6 if pd.isna(r["lap_end"]) else int(r["lap_end"])
                m = (L["driver_number"] == d) & (L["lap_number"] >= ls) & (L["lap_number"] <= le)
                L.loc[m, "Stint"] = r["stint_number"]
                L.loc[m, "Compound"] = "UNKNOWN" if pd.isna(r["compound"]) else str(r["compound"]).upper()
                age0 = 0 if pd.isna(r["tyre_age_at_start"]) else float(r["tyre_age_at_start"])
                L.loc[m, "TyreLife"] = age0 + (L.loc[m, "lap_number"] - ls) + 1
                if i < len(rows) - 1 and not pd.isna(r["lap_end"]):
                    pit_in.add((d, int(r["lap_end"])))
                if i > 0 and not pd.isna(r["lap_start"]):
                    pit_out.add((d, int(r["lap_start"])))
    key = list(zip(L["driver_number"], L["lap_number"].astype(int)))
    is_out = L["is_pit_out_lap"].fillna(False).astype(bool).to_numpy()
    in_mask = np.array([k in pit_in for k in key])
    out_mask = np.array([(k in pit_out) for k in key]) | is_out

    # track status from race control, deleted laps from messages
    wins = status_windows(rc_rows)
    status = []
    for s, e in zip(L["start"], L["end"]):
        codes = sorted({c for c, w0, w1 in wins if pd.notna(s) and pd.notna(e) and s < w1 and e > w0})
        status.append("".join(codes) if codes else "1")
    deleted = set()
    for r in rc_rows or []:
        if "DELETED" in str(r.get("message", "")).upper() and r.get("driver_number") and r.get("lap_number"):
            deleted.add((int(r["driver_number"]), int(r["lap_number"])))

    secs_ok = L[["duration_sector_1", "duration_sector_2", "duration_sector_3"]].notna().all(axis=1) & L["dur"].notna()
    td = lambda col: pd.to_timedelta(pd.to_numeric(L[col], errors="coerce"), unit="s")
    end_td = pd.to_timedelta(L["TimeS"], unit="s")
    out = pd.DataFrame({
        "Driver": L["Driver"], "DriverNumber": L["driver_number"].astype(str), "LapNumber": L["lap_number"].astype(int),
        "LapTime": pd.to_timedelta(L["dur"], unit="s"), "Sector1Time": td("duration_sector_1"),
        "Sector2Time": td("duration_sector_2"), "Sector3Time": td("duration_sector_3"), "Time": end_td,
        "Stint": L["Stint"], "Compound": L["Compound"], "TyreLife": L["TyreLife"], "TrackStatus": status,
        "Deleted": [k in deleted for k in key], "IsAccurate": secs_ok.to_numpy(),
        "PitInTime": end_td.where(in_mask, pd.NaT), "PitOutTime": end_td.where(out_mask, pd.NaT)})
    windows = {(r.Driver, int(r.lap_number)): (r.start, float(r.dur) if pd.notna(r.dur) else np.nan, int(r.driver_number))
               for r in L.itertuples()}
    teams = {(drv.get(n) or {}).get("name_acronym") or str(n): (drv.get(n) or {}).get("team_name", "") for n in drv}
    return out, windows, teams


# ----------------------------------------------------------------------------- telemetry
def refine_start(loc: pd.DataFrame, approx_start: pd.Timestamp, ref_point, ref_dir, search_s: float = 3.0):
    """Time at which this car crosses the reference line (plane through ref_point, normal ref_dir) nearest approx_start.
    Falls back to approx_start. Needed because OpenF1's lap date_start is approximate."""
    if loc is None or len(loc) < 3:
        return approx_start
    t = (loc["t"] - approx_start).dt.total_seconds().to_numpy()
    s = (loc["x"].to_numpy(float) - ref_point[0]) * ref_dir[0] + (loc["y"].to_numpy(float) - ref_point[1]) * ref_dir[1]
    best = None
    for i in range(len(s) - 1):
        if s[i] < 0 <= s[i + 1] and abs(t[i]) <= search_s + 1 and s[i + 1] != s[i]:
            tc = t[i] + (0 - s[i]) * (t[i + 1] - t[i]) / (s[i + 1] - s[i])
            if abs(tc) <= search_s and (best is None or abs(tc) < abs(best)):
                best = tc
    return approx_start if best is None else approx_start + pd.Timedelta(seconds=float(best))


def _loc_frame(loc_rows) -> pd.DataFrame:
    if not loc_rows:
        return pd.DataFrame(columns=["t", "x", "y"])
    d = pd.DataFrame(loc_rows)
    d["t"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
    return d.dropna(subset=["t"]).sort_values("t").drop_duplicates("t")[["t", "x", "y"]].reset_index(drop=True)


def drop_stale_braking_samples(cd: pd.DataFrame, min_run: int = 3, min_change_kmh: float = 15.0):
    """Remove repeated 'stale' speed samples inside a braking zone.

    OpenF1 sometimes repeats the last speed value for about a second. On a straight that looks like a steady speed,
    but while braking (brake on) with the speed clearly different before and after the run it is a data dropout, and
    leaving it in makes integrated distance and the delta-time curve jump. The run's interior is dropped so the
    surrounding samples are interpolated across it. Returns (frame, number of samples dropped)."""
    sp = pd.to_numeric(cd["speed"], errors="coerce").to_numpy(float)
    br = pd.to_numeric(cd["brake"], errors="coerce").to_numpy(float) > 50
    n, drop, i = len(sp), np.zeros(len(sp), bool), 0
    while i < n:
        j = i
        while j + 1 < n and sp[j + 1] == sp[i]:
            j += 1
        if j - i + 1 >= min_run and br[i:j + 1].all() and i > 0 and j < n - 1 and abs(sp[i - 1] - sp[j + 1]) >= min_change_kmh:
            drop[i:j + 1] = True
        i = j + 1
    return (cd.loc[~drop].reset_index(drop=True) if drop.any() else cd), int(drop.sum())


def build_telemetry(cd_rows, loc_rows, t0: pd.Timestamp, duration_s: float) -> pd.DataFrame:
    """OpenF1 car_data/location rows -> FastF1-like lap telemetry (Distance, T, Time, Speed, Throttle, Brake, nGear, X, Y).
    Channels are interpolated exactly at the lap boundaries [0, duration]; Distance is integrated speed."""
    if not cd_rows:
        raise OpenF1Error("OpenF1 returned no car_data for this lap.")
    cd = pd.DataFrame(cd_rows)
    cd["t"] = pd.to_datetime(cd["date"], utc=True, errors="coerce")
    cd = cd.dropna(subset=["t"]).sort_values("t").drop_duplicates("t")
    cd, n_stale = drop_stale_braking_samples(cd)
    ts = (cd["t"] - t0).dt.total_seconds().to_numpy()
    if (ts > 0).sum() < 20 or ts.min() > 1.0 or ts.max() < duration_s - 1.0:
        raise OpenF1Error("car_data does not cover the whole lap (too few samples).")
    inside = ts[(ts > 0.05) & (ts < duration_s - 0.05)]
    grid = np.r_[0.0, inside, float(duration_s)]
    lin = lambda col: np.interp(grid, ts, pd.to_numeric(cd[col], errors="coerce").to_numpy(float))
    idx = np.clip(np.searchsorted(ts, grid, side="right") - 1, 0, len(ts) - 1)
    speed = lin("speed")
    dist = np.r_[0.0, np.cumsum(np.diff(grid) * (speed[1:] + speed[:-1]) / 2 / 3.6)]
    tel = pd.DataFrame({
        "Distance": dist, "T": grid, "Time": pd.to_timedelta(grid, unit="s"), "Speed": speed, "Throttle": lin("throttle"),
        "Brake": pd.to_numeric(cd["brake"], errors="coerce").to_numpy(float)[idx] > 50,
        "nGear": pd.to_numeric(cd["n_gear"], errors="coerce").to_numpy(float)[idx], "RPM": lin("rpm") if "rpm" in cd else np.nan})
    tel.attrs["stale_speed_samples_dropped"] = n_stale
    loc = _loc_frame(loc_rows)
    if len(loc) >= 3:
        lt = (loc["t"] - t0).dt.total_seconds().to_numpy()
        tel["X"], tel["Y"] = np.interp(grid, lt, loc["x"].to_numpy(float)), np.interp(grid, lt, loc["y"].to_numpy(float))
    else:
        tel["X"], tel["Y"] = np.nan, np.nan
    return tel


def corners_from_multiviewer(info: dict, tel: pd.DataFrame | None) -> pd.DataFrame | None:
    """Corner distances along the lap from MultiViewer corner coordinates. Returns None unless the coordinate systems
    clearly match (median nearest-point distance < 5% of the track extent), trying scale 1, 0.1 and 10."""
    corners = (info or {}).get("corners") or []
    if not corners or tel is None or tel[["X", "Y"]].isna().any().any():
        return None
    xy, dist = tel[["X", "Y"]].to_numpy(float), tel["Distance"].to_numpy(float)
    pts = np.array([[c["trackPosition"]["x"], c["trackPosition"]["y"]] for c in corners], float)
    diag = float(np.hypot(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    best = None
    for scale in (1.0, 0.1, 10.0):
        p = pts * scale
        d2 = ((p[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
        idx = d2.argmin(1)
        med = float(np.median(np.sqrt(d2[np.arange(len(p)), idx])))
        if best is None or med < best[0]:
            best = (med, idx)
    if best[0] > 0.05 * diag:
        return None
    return pd.DataFrame({"Number": [int(c.get("number", i + 1)) for i, c in enumerate(corners)],
                         "Letter": [str(c.get("letter") or "") for c in corners], "Distance": dist[best[1]]})


class OpenF1Session:
    """Duck-types the bits of a FastF1 Session that service.compute_all needs."""

    def __init__(self, client, info, laps, windows, teams, meta):
        self.client, self.info, self.laps, self.windows, self.teams, self.meta = client, info, laps, windows, teams, meta
        self._ref = None       # (point, direction) of the reference start line, set by the first telemetry call
        self._ref_tel = None   # first telemetry (driver A) for corner mapping

    def telemetry(self, code: str, lap_number: int) -> pd.DataFrame:
        if (code, lap_number) not in self.windows:
            raise OpenF1Error(f"No lap {lap_number} for {code}.")
        start, dur, num = self.windows[(code, lap_number)]
        if not np.isfinite(dur):
            raise OpenF1Error(f"Lap {lap_number} of {code} has no lap time.")
        sk = self.info["session_key"]
        pad = pd.Timedelta(seconds=3)
        loc_rows = self.client.get("location", session_key=sk, driver_number=num,
                                   filters=[f"date>={_ts(start - pad)}", f"date<={_ts(start + pd.Timedelta(seconds=dur) + pad)}"])
        loc = _loc_frame(loc_rows)
        t0 = start
        if len(loc) >= 5:
            if self._ref is None:
                # first lap defines the reference line: where the car is at its (approximate) lap start
                tsec = (loc["t"] - start).dt.total_seconds().to_numpy()
                x0 = float(np.interp(0.0, tsec, loc["x"].to_numpy(float)))
                y0 = float(np.interp(0.0, tsec, loc["y"].to_numpy(float)))
                ahead = loc[(loc["t"] > start) & (loc["t"] <= start + pd.Timedelta(seconds=1.5))]
                if len(ahead):
                    v = np.array([ahead["x"].iloc[-1] - x0, ahead["y"].iloc[-1] - y0], float)
                    n = np.linalg.norm(v)
                    if n > 0:
                        self._ref = ((x0, y0), tuple(v / n))
            else:
                # later laps: re-anchor the start on the same track line, then use the official lap time as duration
                t0 = refine_start(loc, start, *self._ref)
        t1 = t0 + pd.Timedelta(seconds=dur)
        cd_rows = self.client.get("car_data", session_key=sk, driver_number=num,
                                  filters=[f"date>={_ts(t0 - pd.Timedelta(seconds=1.5))}", f"date<={_ts(t1 + pd.Timedelta(seconds=1.5))}"])
        tel = build_telemetry(cd_rows, loc_rows, t0, dur)
        if self._ref_tel is None:
            self._ref_tel = tel
        return tel

    def get_circuit_info(self):
        url = self.info.get("circuit_info_url")
        if not url:
            return None
        try:
            corners = corners_from_multiviewer(self.client.fetch_json(url), self._ref_tel)
        except Exception:  # best effort only
            return None
        return None if corners is None else SimpleNamespace(corners=corners)


# ----------------------------------------------------------------------------- top level
def load_session(year: int, gp, session_code: str = "R", cache_dir: str = ".openf1_cache", client: OpenF1Client | None = None) -> OpenF1Session:
    client = client or OpenF1Client(cache_dir)
    info = find_session(client, year, gp, session_code)
    sk = info["session_key"]
    laps_rows = client.get("laps", session_key=sk)
    stints_rows = client.get("stints", session_key=sk)
    drivers_rows = client.get("drivers", session_key=sk)
    rc_rows = client.get("race_control", session_key=sk)
    laps, windows, teams = build_laps(laps_rows, stints_rows, drivers_rows, rc_rows)
    meta = dict(event=info["event"], location=info["location"], country=info["country"], session_name=info["session_name"],
                session_date=str(info["date_start"]), fastf1_version="n/a", weather=None, teams=teams, openf1_session_key=sk,
                data_source="OpenF1 API (openf1.org), historical data; derived track status and lap accuracy")
    return OpenF1Session(client, info, laps, windows, teams, meta)


def compute_openf1(cfg, client: OpenF1Client | None = None) -> dict:
    """Full analysis from OpenF1 data (same result structure as service.compute_all)."""
    from .service import compute_all
    s = load_session(cfg.year, cfg.gp, cfg.session, str(Path(cfg.cache_dir) / "openf1"), client)
    meta = dict(s.meta, teams={d: s.teams.get(d, "") for d in (cfg.driver_a, cfg.driver_b)})
    return compute_all(cfg, session=s, tel_fn=lambda sess, drv, lap: sess.telemetry(drv, lap), meta=meta)


def screen_openf1(year: int, gp, session_code: str = "R", cache_dir: str = ".openf1_cache", client: OpenF1Client | None = None) -> pd.DataFrame:
    """Clean-lap counts per driver and best teammate pairs (mirrors pipeline.screen_session)."""
    from . import laps as L
    from .config import AnalysisConfig

    s = load_session(year, gp, session_code, str(Path(cache_dir) / "openf1"), client)
    cfg = AnalysisConfig(year=year, gp=str(gp), session=session_code)
    df = L.flag_clean(L.flag_traffic(L.prepare_laps(s.laps), cfg.traffic_gap_s), cfg)
    stat = df.groupby("Driver").agg(total_laps=("LapNumber", "count"), clean_laps=("Clean", "sum"))
    stat["clear_air_clean"] = df[df["Clean"] & ~df["InTraffic"].fillna(False)].groupby("Driver").size()
    stat = stat.fillna(0).astype(int)
    stat["team"] = stat.index.map(s.teams)
    ts = df["TrackStatus"].astype(str)
    print(f"{s.meta['event']} {year} {s.meta['session_name']}: SC/VSC laps={int(ts.str.contains('[46]', regex=True).sum())}, "
          f"red-flag laps={int(ts.str.contains('5').sum())}")
    pairs = []
    for team, g in stat.groupby("team"):
        if len(g) >= 2:
            g = g.sort_values("clear_air_clean", ascending=False)
            a, b = g.index[0], g.index[1]
            pairs.append(dict(team=team, driver_a=a, driver_b=b, min_clean=int(min(g.loc[a, "clean_laps"], g.loc[b, "clean_laps"])),
                              min_clear_air=int(min(g.loc[a, "clear_air_clean"], g.loc[b, "clear_air_clean"]))))
    print("\nClean laps per driver:\n", stat.sort_values("clean_laps", ascending=False).to_string())
    pr = pd.DataFrame(pairs).sort_values("min_clear_air", ascending=False) if pairs else pd.DataFrame()
    print("\nTeammate pairs (best first):\n", pr.to_string(index=False) if len(pr) else "none")
    return pr
