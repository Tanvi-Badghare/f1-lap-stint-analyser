"""F1 Lap & Stint Performance Analyser - website.

Run locally:   streamlit run app/streamlit_app.py
Deploy:        Streamlit Community Cloud, main file = app/streamlit_app.py (see docs/WEBSITE.md)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works without `pip install -e .` (e.g. on Streamlit Cloud)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from f1analyser import webplots as W  # noqa: E402
from f1analyser.service import list_bundles, load_bundle, save_bundle  # noqa: E402

DATA_DIR = Path(os.environ.get("F1_WEB_DATA", ROOT / "web_data"))

st.set_page_config(page_title="F1 Lap & Stint Analyser", page_icon="🏎️", layout="wide")

METHODS_MD = """
**Sign convention.** `delta = time_B - time_A`. Positive means **A is faster**. On the telemetry delta curve, a rising line means A is gaining.

**Lap cleaning.** Lap 1, in/out laps, non-green track status, deleted laps, `IsAccurate = False` and laps slower than 107 % of the driver's best valid lap are excluded. Traffic is a *proxy*: gap to the car ahead at the timing line below the threshold (this lap or the previous).

**Pace gap.** Laps are paired by lap number and compound. We report the median difference with a bootstrap 95 % CI, the share of laps each driver won, the same on clear-air laps only, and a leave-one-out check on the most extreme lap.

**Telemetry.** Two laps only, resampled to a regular distance grid; B's distance is rescaled to A's length. Brake is on/off (no pressure); car data is about 4-5 Hz, so brake points carry roughly 15-20 m uncertainty.

**Stints.** Slopes of lap time against tyre age are *apparent trends*, shown raw, under assumed fuel-burn benefits, relative to the field median, and on clear-air laps. They are **not** isolated tyre-degradation rates: tyre age, fuel burn, track evolution, traffic and driver management move together.

| Confounder | Partial control used | What remains unknown |
|---|---|---|
| Traffic / dirty air | lap-end gap proxy, clear-air repeat | gap during the lap |
| Fuel load | same lap number, assumed fuel sensitivity | real fuel mass, team differences |
| Track evolution / temperature | pace relative to the field median | the field's own compound and age mix |
| Tyre state | same-compound pairing, tyre age | temperatures, pressures, set history |
| Driver management | slow-lap filter | lift-and-coast, engine modes, gap management |
| 2026 power unit and aero | none | energy deployment/harvesting and active-aero mode are invisible |

**To conclude tyre, braking or setup causes you would need:** fuel mass, tyre temperatures/pressures, brake pressure and steering, ride height and wing levels, engine/energy modes, plus several stints and sessions. Without them, write "consistent with ...", never "caused by ...".
"""


# ----------------------------------------------------------------------------- helpers
def show(fig):
    if fig is None:
        st.info("This chart is not available for this dataset.")
    else:
        st.plotly_chart(fig)


def fmt(x, nd=3, sign=True):
    return "n/a" if x is None or not np.isfinite(x) else (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}")


def headline(a, b, p):
    if p.get("paired_n", 0) < 5:
        base = "Too few like-for-like laps (same lap number AND same tyre compound) for a paired pace comparison. "
        if p.get("unpaired_median_gap") is not None and p.get("unpaired_median_gap") == p.get("unpaired_median_gap"):
            return (base + f"For reference only: across all clean laps the median lap-time difference ({b} minus {a}) is "
                    f"**{fmt(p['unpaired_median_gap'])} s** (95% bootstrap CI {fmt(p.get('unpaired_ci_low'))} to "
                    f"{fmt(p.get('unpaired_ci_high'))} s). The drivers ran different tyres at different times, so this "
                    f"mixes tyre, fuel and track-evolution effects and is **not** a like-for-like pace gap.")
        return base
    faster = a if p["paired_median"] > 0 else b
    return (f"Over {p['paired_n']} same-lap-number, same-compound clean laps, **{faster}** was faster by a median of "
            f"**{abs(p['paired_median']):.3f} s** (95% bootstrap CI {fmt(p['paired_ci_low'])} to {fmt(p['paired_ci_high'])} s, "
            f"{b} minus {a}); {a} was quicker on {100 * p['share_laps_A_faster']:.0f}% of those laps. "
            f"This describes the size and consistency of the gap, not its cause.")


def build_report_md(res, interp: dict) -> str:
    cfg, meta = res["cfg"], res["meta"]
    a, b = cfg["driver_a"], cfg["driver_b"]
    lines = [f"# {meta.get('event', cfg['gp'])} {cfg['year']} - {a} vs {b}",
             f"Session: {meta.get('session_name', cfg['session'])}. Convention: delta = time_{b} - time_{a}; positive means {a} faster.", ""]
    if meta.get("synthetic"):
        lines += ["> **SYNTHETIC DEMO DATA - not real F1 data.**", ""]
    lines.append("## Observations")
    for i, f in enumerate(res["findings"], 1):
        mine = interp.get(f["id"], "").strip() or "_(write your interpretation)_"
        lines += [f"### O{i}. {f['title']}", f"**Evidence.** {f['evidence']}", "", f"**Robustness.** {f['robustness']}", "",
                  f"**Caveat.** {f['caveat']}", "", f"**Needed to go further.** {f['needed']}", "",
                  f"**Interpretation.** {mine}", ""]
    lines += ["## Method and limits", METHODS_MD]
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def cached_bundle(path: str, mtime: float):
    return load_bundle(path)


@st.cache_data(show_spinner=False)
def cached_demo():
    from f1analyser.demo import demo_result
    return demo_result()


# ----------------------------------------------------------------------------- sidebar: choose data
st.sidebar.title("🏎️ F1 Lap & Stint Analyser")
bundles = list_bundles(DATA_DIR)
import importlib.util  # noqa: E402

HAS_FASTF1 = importlib.util.find_spec("fastf1") is not None  # absent on a deployed, saved-data-only site
HAS_LIVE = HAS_FASTF1 or importlib.util.find_spec("requests") is not None
options = (["Saved analyses"] if bundles else []) + (["Run live analysis"] if HAS_LIVE else []) + ["Demo (synthetic data)"]
default_idx = 0 if bundles else options.index("Demo (synthetic data)")  # never open on an empty page
source = st.sidebar.radio("Data", options, index=default_idx,
                          help="Saved analyses are fast and need no internet. Live runs call FastF1.")
res = None

if source == "Saved analyses":
    labels = {p.name: p for p in bundles}
    choice = st.sidebar.selectbox("Analysis", list(labels))
    p = labels[choice]
    res = cached_bundle(str(p), (p / "bundle.json").stat().st_mtime)

elif source == "Demo (synthetic data)":
    res = cached_demo()

else:  # live
    engines = ["OpenF1 (free, includes 2026)"] + (["FastF1 (cached sessions, or after F1 TV login support)"] if HAS_FASTF1 else [])
    engine = st.sidebar.selectbox("Data source", engines)
    use_openf1 = engine.startswith("OpenF1")
    st.sidebar.caption("Needs internet. OpenF1 loads in about a minute and is rate-limited, so results are cached on disk."
                       if use_openf1 else "Needs internet. The first load of a session can take several minutes.")
    year = st.sidebar.number_input("Season", 2018, 2030, 2026)
    gp = st.sidebar.text_input("Grand Prix (name or round number)", "Japan")
    session = st.sidebar.selectbox("Session", ["R", "Q", "S", "SQ"])
    cache_dir = st.sidebar.text_input("Cache folder", ".fastf1_cache")
    if st.sidebar.button("1. Screen session (find a teammate pair)"):
        try:
            if use_openf1:
                from f1analyser.openf1 import screen_openf1 as screen_fn
            else:
                from f1analyser.pipeline import screen_session as screen_fn
            with st.spinner("Loading session..."):
                pairs = screen_fn(int(year), gp, session, cache_dir)
            st.session_state["screen"] = pairs
        except Exception as e:  # noqa: BLE001
            st.session_state["screen"] = None
            st.sidebar.error(f"Could not load the session: {e}")
    if st.session_state.get("screen") is not None and len(st.session_state["screen"]):
        st.sidebar.dataframe(st.session_state["screen"], hide_index=True)
    c1, c2 = st.sidebar.columns(2)
    da = c1.text_input("Driver A", "").upper().strip()
    db = c2.text_input("Driver B", "").upper().strip()
    with st.sidebar.expander("Advanced"):
        allow_yellow = st.checkbox("Allow yellow-flag laps", False)
        traffic_gap = st.number_input("Traffic gap (s)", 0.5, 5.0, 1.5, 0.1)
        tel_mode = st.selectbox("Telemetry laps", ["auto", "matched", "fastest", "explicit"])
        la = st.number_input("Lap A", 1, 100, 20)
        lb = st.number_input("Lap B", 1, 100, 20)
    if st.sidebar.button("2. Run analysis", type="primary", disabled=not (da and db)):
        try:
            from f1analyser.config import AnalysisConfig
            if use_openf1:
                from f1analyser.openf1 import compute_openf1 as compute_all
            else:
                from f1analyser.service import compute_all
            cfg = AnalysisConfig(year=int(year), gp=gp, session=session, driver_a=da, driver_b=db, cache_dir=cache_dir,
                                 allow_yellow=allow_yellow, traffic_gap_s=traffic_gap,
                                 tel_mode="matched" if tel_mode == "explicit" else tel_mode,
                                 tel_laps=(int(la), int(lb)) if tel_mode == "explicit" else None)
            with st.spinner("Analysing (loading telemetry can take a few minutes)..."):
                st.session_state["live"] = compute_all(cfg)
        except Exception as e:  # noqa: BLE001
            st.sidebar.error(f"Analysis failed: {e}")
    res = st.session_state.get("live")
    if res is not None and st.sidebar.button("Save to website folder"):
        out = save_bundle(res, DATA_DIR)
        st.sidebar.success(f"Saved to {out}. Commit this folder to publish it.")

if res is None:
    st.title("F1 Lap & Stint Performance Analyser")
    st.info("Pick a data source in the sidebar. Choose **Run live analysis** to analyse a real session, "
            "or **Demo** to explore the site with synthetic data.")
    st.stop()

# ----------------------------------------------------------------------------- unpack
cfg, meta, fr, pace = res["cfg"], res["meta"], res["frames"], res["pace"]
a, b = cfg["driver_a"], cfg["driver_b"]
laps = fr["laps"]
clean = laps[laps["Clean"]]
tl = res["tel_laps"]

st.title(f"{meta.get('event', cfg['gp'])} {cfg['year']} - {meta.get('session_name', cfg['session'])}")
teams = meta.get("teams") or {}
st.caption(f"**{a}** {teams.get(a, '')} (A, blue) vs **{b}** {teams.get(b, '')} (B, orange). "
           f"Delta = {b} - {a}; positive = {a} faster. Source: {str(meta.get('data_source', 'FastF1')).rstrip('.')}.")
if meta.get("synthetic"):
    st.warning("SYNTHETIC DEMO DATA: generated for testing the site. These are not real F1 results.")

tabs = st.tabs(["Summary", "Laps & sectors", "Telemetry", "Stints", "Observations", "Method & limits", "Data"])

# ---- Summary
with tabs[0]:
    st.markdown(headline(a, b, pace))
    c = st.columns([1, 2.2, 1.2, 1.2, 1])
    c[0].metric("Median paired gap", f"{fmt(pace.get('paired_median'))} s", help=f"{b} minus {a}; positive = {a} faster")
    c[1].metric("95% bootstrap CI", f"{fmt(pace.get('paired_ci_low'))} to {fmt(pace.get('paired_ci_high'))} s")
    c[2].metric(f"Laps won by {a}", f"{100 * pace.get('share_laps_A_faster', float('nan')):.0f}%" if pace.get("paired_n") else "n/a")
    c[3].metric("Best-lap gap", f"{fmt(pace.get('best_lap_gap'))} s")
    c[4].metric("Clean laps", f"{int((clean['Driver'] == a).sum())} / {int((clean['Driver'] == b).sum())}", help=f"{a} / {b}")
    st.caption(f"Clear-air median gap: {fmt(pace.get('clear_median'))} s (n={pace.get('clear_n', 0)}). "
               f"Without the single most extreme lap: {fmt(pace.get('median_without_most_extreme_lap'))} s. "
               f"Top-5 mean gap: {fmt(pace.get('top5_mean_gap'))} s.")
    show(W.lap_times_fig(laps, a, b))
    if len(fr["paired"]):
        show(W.paired_delta_fig(fr["paired"], pace, a, b))

# ---- Laps & sectors
with tabs[1]:
    st.subheader("Best laps and consistency")
    st.dataframe(fr["best_laps"], hide_index=True)
    st.dataframe(fr["lap_stats"], hide_index=True)
    st.subheader("Sectors")
    show(W.sector_fig(fr["sectors"], a, b))
    st.dataframe(fr["sectors"], hide_index=True)
    st.dataframe(fr["theoretical_best"], hide_index=True)

# ---- Telemetry
with tabs[2]:
    if not tl:
        st.info(meta.get("telemetry_note", "No telemetry in this analysis.") + " Lap, sector and stint tabs are unaffected.")
    else:
        st.info(f"Compared laps: {a} lap {tl['lap_a']} vs {b} lap {tl['lap_b']} (`{tl['mode']}`): {tl['reason']}. "
                f"Alignment: {res['tel_info'].get('alignment', 'n/a')}; distance of {b} rescaled by {res['tel_info']['b_distance_scale']:.4f}.")
        corners = fr["corners"] if len(fr["corners"]) else None
        show(W.telemetry_fig(fr["aligned"], a, b, corners))
        c1, c2 = st.columns(2)
        with c1:
            show(W.dominance_fig(fr["aligned"], fr["minis"], a, b))
        with c2:
            show(W.minisector_fig(fr["minis"], a, b))
        if corners is not None:
            st.subheader("Corner detail")
            st.dataframe(fr["corners"], hide_index=True)
        st.subheader("Lap-level channel summary")
        st.dataframe(fr["channel_summary"], hide_index=True)
        st.caption("Brake is on/off only. Two laps cannot show whether the pattern repeats: re-run with different laps to check.")

# ---- Stints
with tabs[3]:
    st.warning("Slopes below are **apparent lap-time trends**, not tyre-degradation rates. Tyre age, fuel burn, track evolution, "
               "traffic and driver management are confounded.")
    if len(fr["stint_table"]):
        show(W.stints_fig(laps, fr["stint_table"], [a, b]))
        st.dataframe(fr["stint_table"], hide_index=True)
        if len(fr["fuel_sensitivity"]):
            show(W.fuel_fig(fr["fuel_sensitivity"]))
    else:
        st.info("No stint had enough clean laps (minimum 6) for a trend.")
    show(W.relpace_fig(laps, a, b))

# ---- Observations
with tabs[4]:
    st.caption("Auto-drafted from the numbers. Keep the 3-5 you can defend and write your own interpretation under each.")
    interp = {}
    for i, f in enumerate(res["findings"], 1):
        with st.expander(f"O{i}. {f['title']}", expanded=(i == 1)):
            st.markdown(f"**Evidence.** {f['evidence']}")
            st.markdown(f"**Robustness.** {f['robustness']}")
            st.markdown(f"**Caveat.** {f['caveat']}")
            st.markdown(f"**Needed to go further.** {f['needed']}")
            interp[f["id"]] = st.text_area("Your interpretation", key=f"interp_{f['id']}", height=100,
                                           placeholder="What do the plots show? What else could explain it? What would change your mind?")
    st.download_button("Download report (Markdown)", build_report_md(res, interp),
                       file_name=f"{a}-vs-{b}_{cfg['year']}_{cfg['gp']}.md", mime="text/markdown")

# ---- Method & limits
with tabs[5]:
    st.markdown(METHODS_MD)
    exc = fr["exclusion_summary"]
    st.subheader("Why laps were excluded")
    st.dataframe(exc, hide_index=True)

# ---- Data
with tabs[6]:
    names = [k for k, v in fr.items() if isinstance(v, pd.DataFrame) and len(v)]
    pick = st.selectbox("Table", names)
    st.dataframe(fr[pick], hide_index=True)
    st.download_button("Download CSV", fr[pick].to_csv(index=False), file_name=f"{pick}.csv", mime="text/csv")
    with st.expander("Run metadata"):
        st.json({k: res[k] for k in ("meta", "cfg", "tel_laps", "tel_info")}, expanded=False)
