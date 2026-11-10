"""FastF1 data access. This is the only module (with cli.py) that touches the network."""
from __future__ import annotations

from pathlib import Path


def load_session(year: int, gp, session: str, cache_dir: str = ".fastf1_cache", telemetry: bool = True):
    """Load a FastF1 session, caching raw API responses on disk (first load can take minutes)."""
    import fastf1

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    try:
        event = int(gp)
    except (TypeError, ValueError):
        event = gp
    s = fastf1.get_session(int(year), event, session)
    s.load(laps=True, telemetry=telemetry, weather=True, messages=False)
    try:
        _ = s.laps
    except Exception as e:
        raise RuntimeError(
            "FastF1 could not load lap data. Since about 15 Oct 2026 F1's timing archive answers HTTP 401 "
            "(F1 TV login required), so FastF1 cannot load sessions that are not already in its cache. "
            "Use the default OpenF1 source instead (omit --source fastf1). Other causes: a wrong event name/year, "
            "a session that has not taken place, or an interrupted cache download (delete the cache folder).") from e
    return s


def session_metadata(session) -> dict:
    """Provenance block written next to every result."""
    import fastf1
    import numpy as np
    import pandas as pd
    import scipy
    import matplotlib

    meta = {
        "fastf1_version": fastf1.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "event": str(session.event.get("EventName", "")),
        "location": str(session.event.get("Location", "")),
        "country": str(session.event.get("Country", "")),
        "session_name": str(session.name),
        "session_date": str(session.date),
        "data_source": "F1 live-timing API via FastF1 (see docs/DATA_SOURCE.md)",
    }
    try:
        w = session.weather_data
        meta["weather"] = {
            "air_temp_mean_c": float(w["AirTemp"].mean()),
            "track_temp_mean_c": float(w["TrackTemp"].mean()),
            "track_temp_range_c": [float(w["TrackTemp"].min()), float(w["TrackTemp"].max())],
            "any_rainfall": bool(w["Rainfall"].any()),
        }
    except Exception:  # weather is optional
        meta["weather"] = None
    return meta


def driver_team(session, code: str) -> str:
    try:
        return str(session.get_driver(code)["TeamName"])
    except Exception:
        return ""
