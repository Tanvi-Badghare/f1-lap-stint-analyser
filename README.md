# F1 Lap & Stint Performance Analyser

Compare two drivers from **one Grand Prix session**: lap and sector pace, distance-aligned telemetry, stint trends,
and a draft engineering report whose claims are bounded by what public data can support. It never claims isolated
tyre degradation or a car-setup cause (a test enforces the wording).

> Not affiliated with Formula 1. This repository contains **code only, no F1 data**. See
> [docs/DATA_POLICY.md](docs/DATA_POLICY.md) before sharing any data you download.

## Project status (updated 5 Nov 2026)
| Part | Status |
|---|---|
| Statistics, lap cleaning, stints, telemetry alignment, findings | Done; tested on synthetic data with known answers |
| Streamlit website (7 tabs, Plotly) | Done; checked on the synthetic demo |
| FastF1 data source | Code complete, but F1's timing archive has required a login (HTTP 401) since about 15 Oct 2026, so uncached sessions fail |
| OpenF1 data source (`openf1.py`) | Laps, stints, drivers and race control read from real 2026 Japan data. `location`/`car_data` now answer 401 without an account, so telemetry is skipped with a note |
| Full two-driver analysis on real 2026 data | Runs end to end on Japan 2026 (PIA vs NOR, one race, one lap pair). Lap/sector arithmetic checks pass; the telemetry delta curve matches the official sector-split differences within 0.01 s and the final lap-time difference exactly. Telemetry positions inside braking zones carry about 10-20 m uncertainty and OpenF1 speed data can contain stale repeated samples (repaired and counted). Not validated on other races |
| All 2026 rounds (`build_season`), weekly updates, predictor | Planned, not built |

36 tests, no network needed. Roadmap and theory: see the technical report (docs/).

## What it produces
`outputs/<year>_<gp>_<session>_<A>-vs-<B>/`
- `report_draft.md`: computed evidence + prompts where **you** write the interpretation
- `figures/*.png`: lap distribution, sector deltas, paired lap delta, telemetry (speed / delta / throttle / brake / gear),
  dominance map, mini-sector bars, stints, field-relative pace, fuel-assumption sensitivity
- `tables/*.csv`: every number in the report (exclusions, best laps, sectors, stints, corners ...)
- `findings.json`, `run_metadata.json` (versions, parameters, lap-selection reasons: full provenance)

## Setup
```bash
git clone <your-repo-url> && cd f1-lap-stint-analyser
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest -q                                     # 36 tests on synthetic data and a simulated OpenF1 server, no internet needed
```

## Run
**Website (recommended, OpenF1 source, includes 2026):**
```bash
pip install -e ".[web]"
python -m f1analyser.build_web --year 2026 --gp Japan --screen          # list teammate pairs and clean-lap counts
python -m f1analyser.build_web --year 2026 --gp Japan --drivers PIA NOR # add --no-telemetry if OpenF1 refuses car data
python -m streamlit run app/streamlit_app.py                            # choose "Saved analyses"
python -m f1analyser.build_web --demo                                   # SYNTHETIC demo bundle, no network
```
OpenF1 now requires an account for position and car data (telemetry). Set `OPENF1_USERNAME` and `OPENF1_PASSWORD` to use one;
without it the lap, sector and stint analysis still works and the Telemetry tab explains why it is empty.

**Command line with FastF1 (older seasons, or sessions already cached):**
```bash
python -m f1analyser screen --year 2024 --gp Bahrain --session R
python -m f1analyser analyse --year 2024 --gp Bahrain --session R --drivers VER PER
```
Since about 15 Oct 2026 F1's timing archive returns HTTP 401, so FastF1 cannot load sessions that are not already cached.
`screen` helps pick a session with enough clean, clear-air laps; teammates are the cleanest comparison (same car).

## Method in six lines
1. **Clean laps**: drop lap 1, in/out laps, non-green track status, deleted laps, FastF1 `IsAccurate=False`, laps > 107 % of the driver's best valid lap. Every exclusion is counted in `tables/exclusion_summary.csv`.
2. **Traffic proxy**: gap to the car ahead at the timing line < 1.5 s (this lap or previous).
3. **Pace gap**: laps paired by lap number and compound; median delta with bootstrap 95 % CI, share of laps won, clear-air-only repeat, leave-one-out check, best-lap vs top-5 vs median gap.
4. **Telemetry**: two laps (matched pair on same compound in clear air, else fastest, reason logged) resampled to 5 m; B's distance rescaled to A's length; delta-time curve; 25 mini-sectors; per-corner brake point, minimum speed, full-throttle point, entry/exit time gain.
5. **Stints**: OLS slope of lap time vs tyre age with SE, repeated (a) raw, (b) under assumed fuel benefits 0-0.04 s/lap, (c) relative to the field median, (d) clear-air laps only. These are *apparent trends*, not tyre degradation rates.
6. **Findings**: generated with evidence, robustness, caveat, and "data needed" for each; you keep 3-5 and rewrite them.

## Layout
```
src/f1analyser/   data.py openf1.py laps.py stints.py telemetry.py stats.py findings.py service.py webplots.py plots.py report.py pipeline.py cli.py build_web.py build_season.py demo.py
app/              streamlit_app.py (website)
tests/            synthetic-data tests (known gap, known slope, known time-loss zone)
docs/             DATA_SOURCE.md  DATA_SOURCE_OPENF1.md  DATA_POLICY.md  METHODS_AND_LIMITS.md  REPORT_GUIDE.md  WEBSITE.md
```

## Limitations (read before quoting results)
No fuel mass, tyre temperature/pressure, brake pressure, steering, ride height or engine mode in public data.
FastF1 `Brake` is on/off. Telemetry is ~4-5 Hz. Results describe one session and two drivers.

## Website
```bash
python -m f1analyser.build_web --year 2026 --gp Japan --screen              # OpenF1 (default source)
python -m f1analyser.build_web --year 2026 --gp Japan --drivers PER BOT
python -m streamlit run app/streamlit_app.py                                # choose "Saved analyses"
python -m f1analyser.build_web --demo                                       # SYNTHETIC demo bundle
```
