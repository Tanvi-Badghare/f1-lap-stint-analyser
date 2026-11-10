# Website (Streamlit)

Interactive version of the analysis: hover on every chart, switch drivers/laps, edit your interpretation under each
observation and download the report as Markdown.

## Files added
```
app/streamlit_app.py          the website
app/requirements.txt          minimal packages for deployment (no FastF1 needed)
.streamlit/config.toml
src/f1analyser/service.py     compute_all(), save_bundle(), load_bundle()
src/f1analyser/webplots.py    interactive Plotly charts
src/f1analyser/demo.py        SYNTHETIC demo data (not real F1 data)
src/f1analyser/build_web.py   builds a data bundle from a real session
tests/test_service_app.py     tests (no internet needed)
```

## Run locally (Windows PowerShell, venv active, Python 3.12 or 3.13)
```powershell
python -m pip install -r app/requirements.txt
python -m pytest -q                     # now 20 tests
python -m streamlit run app/streamlit_app.py
```
It opens at http://localhost:8501. With no saved data it opens on the **Demo** (synthetic) so you can check the site works.

## Use real data
Option 1 - inside the site: choose **Run live analysis**, enter 2026 / Japan / R, click *Screen session*,
type the two driver codes, click *Run analysis*, then *Save to website folder*.

Option 2 - from the terminal (same thing, no browser):
```powershell
python -m f1analyser.build_web --year 2026 --gp Japan --session R --drivers AAA BBB
```
Both write `web_data/2026_japan_R_AAA-vs-BBB/` (small CSV files + `bundle.json`). Restart or refresh the site and pick it
under **Saved analyses**. Use `--tel-laps 20 20` to choose telemetry laps.

## Publish it (free): Streamlit Community Cloud
1. Commit `web_data/` and push the repo to GitHub (the data folder is small; the raw FastF1 cache stays git-ignored).
2. Go to https://share.streamlit.io, sign in with GitHub, **Create app**.
3. Repository: yours; branch: `main`; main file path: `app/streamlit_app.py`.
4. Advanced settings: choose Python **3.12**.
5. Deploy. Streamlit installs `app/requirements.txt` automatically. The deployed site shows **Saved analyses** only,
   because live FastF1 loading is too slow and memory-hungry for a free host. Run live analyses on your computer and
   commit the bundles.

## Notes
- The site shows apparent trends and "consistent with" language only. It does not claim to isolate tyre degradation or setup.
- Bundles hold derived numbers (lap times, resampled telemetry). Check F1 data terms before publishing widely; see `DATA_SOURCE.md`.
- If you change `src/` while the site is running, stop and restart it.
