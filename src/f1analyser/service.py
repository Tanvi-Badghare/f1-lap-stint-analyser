"""In-memory analysis for the website, plus a tiny CSV+JSON "bundle" format.

compute_all()  runs the same lap/stint/telemetry analysis as the CLI but returns DataFrames instead of files.
save_bundle()  writes a result to  <root>/<slug>/  (CSV + bundle.json) so the website can show it without
               FastF1 or internet (this is what you commit and deploy).
load_bundle()  reads it back.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import telemetry as T
from .config import AnalysisConfig
from .findings import build_findings
from .pipeline import analyse_laps, _jsonable

LAP_COLS = ["Driver", "LapNumber", "LapTimeS", "S1S", "S2S", "S3S", "TimeS", "Stint", "Compound", "TyreLife",
            "TyreAge", "TrackStatus", "IsAccurate", "Deleted", "GapAheadS", "MinGapS", "InTraffic",
            "ExclusionReason", "Clean", "FieldMedianS", "RelPaceS"]
BOOL_COLS = ("Clean", "InTraffic", "Deleted", "IsAccurate")
JSON_KEYS = ("meta", "cfg", "tel_laps", "tel_info", "pace", "findings")
ENERGY_NOTE = (" For 2026-regulation cars, electrical energy deployment/harvesting and the active-aero mode also "
               "shape speed traces and are not visible in public data.")


class _NoTelemetry(Exception):
    pass


def slug_of(cfg: dict) -> str:
    gp = str(cfg["gp"]).lower().replace(" ", "-")
    return f"{cfg['year']}_{gp}_{cfg['session']}_{cfg['driver_a']}-vs-{cfg['driver_b']}"


def compute_all(cfg: AnalysisConfig, session=None, tel_fn=None, meta: dict | None = None,
                corners_df: pd.DataFrame | None = None) -> dict:
    """Run the full analysis in memory. ``session``/``tel_fn``/``meta``/``corners_df`` can be injected for tests/demo."""
    if session is None:
        from .data import load_session
        session = load_session(cfg.year, cfg.gp, cfg.session, cfg.cache_dir, telemetry=True)
    if meta is None:
        from .data import session_metadata, driver_team
        meta = session_metadata(session)
        meta["teams"] = {cfg.driver_a: driver_team(session, cfg.driver_a), cfg.driver_b: driver_team(session, cfg.driver_b)}
    tel_fn = tel_fn or T.get_lap_telemetry
    a, b = cfg.driver_a, cfg.driver_b

    res = analyse_laps(session.laps, cfg)
    pair_in = res.clean[res.clean["Driver"].isin([a, b])]
    mode = cfg.tel_mode
    if mode == "auto":
        mode = "fastest" if cfg.session.upper().startswith(("Q", "SQ", "SS")) else "matched"
    tel_note = None
    al, minis, tel_info = pd.DataFrame(), pd.DataFrame(), {}
    try:
        if not cfg.use_telemetry:
            raise _NoTelemetry("Telemetry was switched off (--no-telemetry).")
        tel_laps = T.select_comparison_laps(pair_in, a, b, mode=mode, explicit=cfg.tel_laps)
        ta = tel_fn(session, a, tel_laps["lap_a"])
        tb = tel_fn(session, b, tel_laps["lap_b"])
        al, tel_info = T.align_laps(ta, tb, cfg.grid_step_m)
        minis = T.minisector_deltas(al, cfg.n_minisectors)
    except _NoTelemetry as e:
        tel_note = str(e)
    except RuntimeError as e:
        # e.g. OpenF1Error (HTTP 401/403 on car_data/location): keep the lap/sector/stint analysis
        if type(e).__name__ != "OpenF1Error":
            raise
        tel_note = f"Telemetry could not be loaded from the data source and was skipped: {e}"
    if tel_note:
        tel_laps, al, minis, tel_info = {}, pd.DataFrame(), pd.DataFrame(), {}
        corners_df = None
        meta = dict(meta, telemetry_note=tel_note)

    corners = None
    try:
        if tel_note:
            raise _NoTelemetry(tel_note)
        if corners_df is None:
            ci = session.get_circuit_info()
            corners_df = ci.corners if ci is not None else None
        corners = T.normalise_corners(corners_df, tel_info["length_a_m"])
    except Exception:  # corner annotations are community data and may be missing
        corners = None
    corner_t = (T.corner_table(al, corners, a, b, cfg.corner_before_m, cfg.corner_after_m, cfg.full_throttle_pct)
                if corners is not None and len(corners) else pd.DataFrame())

    tb_ = res.tables
    findings = build_findings(a, b, res.pace, tb_["sectors"], tb_["stint_table"], tb_["fuel_sensitivity"], minis,
                              corner_t if len(corner_t) else None, tel_info, tel_laps)
    if tel_note:
        for f in findings:
            if f["id"] == "limits":
                f["evidence"] += " " + tel_note + " No telemetry findings are made."
    if cfg.year >= 2026:
        for f in findings:
            if f["id"] == "limits":
                f["evidence"] += ENERGY_NOTE

    laps = res.all_laps[res.all_laps["Driver"].isin([a, b])].merge(
        res.clean[["Driver", "LapNumber", "FieldMedianS", "RelPaceS"]], on=["Driver", "LapNumber"], how="left")
    laps = laps[[c for c in LAP_COLS if c in laps.columns]].reset_index(drop=True)

    frames = dict(
        laps=laps, paired=res.paired, aligned=al, minis=minis, corners=corner_t,
        sectors=tb_["sectors"], best_laps=tb_["best_laps"], lap_stats=tb_["lap_stats"],
        theoretical_best=tb_["theoretical_best"], stint_table=tb_["stint_table"],
        fuel_sensitivity=tb_["fuel_sensitivity"], exclusion_summary=tb_["exclusion_summary"],
        compound_comparison=tb_["compound_comparison"],
        channel_summary=(T.channel_summary(al, a, b, cfg.full_throttle_pct) if len(al) else pd.DataFrame()),
    )
    return dict(meta=meta, cfg=cfg.to_dict(), tel_laps=tel_laps, tel_info=tel_info, pace=res.pace,
                findings=findings, frames=frames)


# ----------------------------------------------------------------------------- bundle I/O
def save_bundle(result: dict, root) -> Path:
    out = Path(root) / slug_of(result["cfg"])
    out.mkdir(parents=True, exist_ok=True)
    for name, df in result["frames"].items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(out / f"{name}.csv", index=False)
    payload = {k: result[k] for k in JSON_KEYS}
    (out / "bundle.json").write_text(json.dumps(payload, indent=2, default=_jsonable))
    return out


def load_bundle(path) -> dict:
    p = Path(path)
    result = json.loads((p / "bundle.json").read_text())
    frames = {}
    for f in p.glob("*.csv"):
        frames[f.stem] = pd.read_csv(f)
    laps = frames.get("laps")
    if laps is not None:
        for c in BOOL_COLS:
            if c in laps:
                laps[c] = laps[c].fillna(False).astype(bool)
        laps["ExclusionReason"] = laps["ExclusionReason"].fillna("")
    al = frames.get("aligned")
    if al is not None and len(al):
        for c in ("Brake_A", "Brake_B"):
            al[c] = al[c].astype(bool)
    pr = frames.get("paired")
    if pr is not None and "EitherInTraffic" in pr:
        pr["EitherInTraffic"] = pr["EitherInTraffic"].fillna(False).astype(bool)
    for name in ("aligned", "minis", "channel_summary", "corners", "stint_table", "fuel_sensitivity", "compound_comparison", "paired"):
        frames.setdefault(name, pd.DataFrame())
    result["frames"] = frames
    return result


def list_bundles(root) -> list[Path]:
    r = Path(root)
    if not r.exists():
        return []
    return sorted(p.parent for p in r.glob("*/bundle.json"))
