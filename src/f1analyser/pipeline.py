"""Orchestration: pure lap/stint analysis (testable) + telemetry + file outputs."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import laps as L
from . import stints as S
from . import telemetry as T
from . import plots as P
from .config import AnalysisConfig
from .findings import build_findings
from .report import render_report


@dataclass
class LapResult:
    all_laps: pd.DataFrame
    clean: pd.DataFrame
    tables: dict = field(default_factory=dict)
    paired: pd.DataFrame | None = None
    pace: dict = field(default_factory=dict)


def analyse_laps(raw_laps: pd.DataFrame, cfg: AnalysisConfig) -> LapResult:
    """Everything that needs only the lap table (no telemetry, no network)."""
    a, b = cfg.driver_a, cfg.driver_b
    df = L.prepare_laps(raw_laps)
    df = L.flag_traffic(df, cfg.traffic_gap_s)
    df = L.flag_clean(df, cfg)
    df = S.add_tyre_age(df)
    clean = df[df["Clean"]].copy()
    ref = S.field_reference(clean, cfg.min_cars_for_field_ref)
    clean = S.add_relative_pace(clean, ref)
    drivers = [a, b]
    for d in drivers:
        n = int((clean["Driver"] == d).sum())
        if n < 10:
            print(f"WARNING: only {n} clean laps for {d}; conclusions will be weak.", file=sys.stderr)

    pair_in = clean[clean["Driver"].isin(drivers)]
    paired = L.paired_lap_delta(pair_in, a, b, cfg.paired_same_compound)
    pace = L.pace_gap_summary(paired, pair_in, a, b, cfg)
    stint_t = S.stint_table(clean, drivers, cfg)
    t = dict(
        exclusion_summary=L.exclusion_summary(df, drivers),
        best_laps=L.best_laps(clean, drivers),
        lap_stats=L.lap_stats(clean, drivers),
        sectors=L.sector_table(pair_in, a, b),
        theoretical_best=L.theoretical_best(clean, drivers),
        paired_laps=paired,
        stint_table=stint_t,
        fuel_sensitivity=S.fuel_sensitivity(clean, drivers, cfg),
        compound_comparison=S.compound_comparison(stint_t, a, b) if len(stint_t) else pd.DataFrame(),
    )
    return LapResult(all_laps=df, clean=clean, tables=t, paired=paired, pace=pace)


def _write_tables(tables: dict, out: Path):
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for k, v in tables.items():
        if isinstance(v, pd.DataFrame) and not v.empty:
            v.to_csv(out / "tables" / f"{k}.csv", index=False)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def run_analysis(cfg: AnalysisConfig, command: str = "") -> Path:
    from .data import load_session, session_metadata, driver_team

    session = load_session(cfg.year, cfg.gp, cfg.session, cfg.cache_dir, telemetry=True)
    meta = session_metadata(session)
    meta["command"] = command
    meta["teams"] = {cfg.driver_a: driver_team(session, cfg.driver_a), cfg.driver_b: driver_team(session, cfg.driver_b)}
    out = Path(cfg.output_dir) / cfg.slug()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    a, b = cfg.driver_a, cfg.driver_b

    res = analyse_laps(session.laps, cfg)
    tables, clean = res.tables, res.clean
    drivers = [a, b]
    figs = {}
    pair_in = clean[clean["Driver"].isin(drivers)]

    # ---- lap / sector / stint figures
    figs["lap_distribution"] = P.plot_lap_distribution(pair_in, a, b, out / "figures/lap_distribution.png")
    figs["sector_deltas"] = P.plot_sector_deltas(tables["sectors"], a, b, out / "figures/sector_deltas.png")
    if len(res.paired):
        figs["paired_delta"] = P.plot_paired_delta(res.paired, res.pace, a, b, out / "figures/paired_delta.png")
    if len(tables["stint_table"]):
        figs["stints"] = P.plot_stints(clean, res.all_laps, drivers, a, tables["stint_table"], out / "figures/stints.png")
        figs["fuel_sensitivity"] = P.plot_fuel_sensitivity(tables["fuel_sensitivity"], out / "figures/fuel_sensitivity.png")
    figs["relative_pace"] = P.plot_relative_pace(clean, drivers, a, out / "figures/relative_pace.png")

    # ---- telemetry
    tel_laps = T.select_comparison_laps(
        pair_in, a, b, mode=("fastest" if cfg.tel_mode == "auto" and cfg.session.upper().startswith(("Q", "SQ", "SS"))
                              else ("matched" if cfg.tel_mode == "auto" else cfg.tel_mode)),
        explicit=cfg.tel_laps)
    print(f"Telemetry laps: {a} lap {tel_laps['lap_a']}, {b} lap {tel_laps['lap_b']} ({tel_laps['mode']})")
    ta = T.get_lap_telemetry(session, a, tel_laps["lap_a"])
    tb = T.get_lap_telemetry(session, b, tel_laps["lap_b"])
    al, tel_info = T.align_laps(ta, tb, cfg.grid_step_m)
    if abs(tel_info["b_distance_scale"] - 1) > 0.01:
        print(f"WARNING: lap distances differ by {100 * (tel_info['b_distance_scale'] - 1):+.2f}%; check the laps.", file=sys.stderr)
    minis = T.minisector_deltas(al, cfg.n_minisectors)
    corners = None
    try:
        ci = session.get_circuit_info()
        corners = T.normalise_corners(ci.corners if ci is not None else None, tel_info["length_a_m"])
    except Exception as e:  # circuit info is community-maintained and can be missing
        print(f"NOTE: no corner annotations ({e}); corner table skipped.", file=sys.stderr)
    corner_t = T.corner_table(al, corners, a, b, cfg.corner_before_m, cfg.corner_after_m, cfg.full_throttle_pct) \
        if corners is not None else pd.DataFrame()
    tables.update(minisectors=minis, corners=corner_t, channel_summary=T.channel_summary(al, a, b, cfg.full_throttle_pct))
    info_txt = f"{a} lap {tel_laps['lap_a']} vs {b} lap {tel_laps['lap_b']} ({tel_laps['mode']})"
    figs["telemetry"] = P.plot_telemetry(al, a, b, corners, out / "figures/telemetry.png", info_txt)
    figs["dominance_map"] = P.plot_dominance_map(al, minis, a, b, out / "figures/dominance_map.png")
    figs["minisector_bars"] = P.plot_minisector_bars(minis, a, b, out / "figures/minisector_bars.png")

    # ---- findings + report
    findings = build_findings(a, b, res.pace, tables["sectors"], tables["stint_table"], tables["fuel_sensitivity"],
                              minis, corner_t if len(corner_t) else None, tel_info, tel_laps)
    _write_tables(tables, out)
    (out / "findings.json").write_text(json.dumps(findings, indent=2, default=_jsonable))
    meta.update(config=cfg.to_dict(), telemetry_laps=tel_laps, telemetry_alignment=tel_info, pace_summary=res.pace)
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=_jsonable))
    (out / "report_draft.md").write_text(render_report(cfg, meta, tables, figs, findings, tel_laps, tel_info))
    print(f"Done. Outputs in {out}")
    return out


def screen_session(year: int, gp, session: str, cache_dir: str, cfg: AnalysisConfig | None = None) -> pd.DataFrame:
    """Rank teammate pairs by clean-lap count so the session/pair is chosen with evidence, not hope."""
    from .data import load_session

    s = load_session(year, gp, session, cache_dir, telemetry=False)
    cfg = cfg or AnalysisConfig(year=year, gp=str(gp), session=session)
    df = L.flag_clean(L.flag_traffic(L.prepare_laps(s.laps), cfg.traffic_gap_s), cfg)
    stat = df.groupby("Driver").agg(total_laps=("LapNumber", "count"), clean_laps=("Clean", "sum"),
                                    clear_air_clean=("InTraffic", lambda x: 0)).drop(columns="clear_air_clean")
    stat["clear_air_clean"] = df[df["Clean"] & ~df["InTraffic"].fillna(False)].groupby("Driver").size()
    stat["stints"] = df[df["Clean"]].groupby("Driver")["Stint"].nunique()
    stat = stat.fillna(0).astype(int)
    try:
        team = s.results.set_index("Abbreviation")["TeamName"]
        stat["team"] = stat.index.map(team)
    except Exception:
        stat["team"] = ""
    ts = df["TrackStatus"].astype(str)
    summary = dict(sc_or_vsc_laps=int(ts.str.contains("[467]", regex=True).sum()), red_flag_laps=int(ts.str.contains("5").sum()),
                   rain=bool(s.weather_data["Rainfall"].any()) if s.weather_data is not None else None)
    print(f"{s.event['EventName']} {year} {s.name}: {summary}")
    pairs = []
    for team, g in stat.groupby("team"):
        if len(g) >= 2:
            g = g.sort_values("clear_air_clean", ascending=False)
            x, y = g.index[0], g.index[1]
            pairs.append(dict(team=team, driver_a=x, driver_b=y, min_clean=int(min(g.loc[x, "clean_laps"], g.loc[y, "clean_laps"])),
                              min_clear_air=int(min(g.loc[x, "clear_air_clean"], g.loc[y, "clear_air_clean"]))))
    print("\nClean laps per driver:\n", stat.sort_values("clean_laps", ascending=False).to_string())
    pr = pd.DataFrame(pairs).sort_values("min_clear_air", ascending=False) if pairs else pd.DataFrame()
    print("\nTeammate pairs (best first):\n", pr.to_string(index=False) if len(pr) else "none")
    return pr
