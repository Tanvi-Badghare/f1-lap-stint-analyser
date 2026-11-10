# Data source

- **Library:** [FastF1](https://docs.fastf1.dev) (version recorded in every `run_metadata.json`).
- **Upstream:** Formula 1 live-timing / timing-data feeds (laps, sectors, tyres, track status, car data, positions, weather);
  session results and driver lists via Jolpica/Ergast; track-corner annotations via the community-maintained MultiViewer circuit API.
- **Access:** downloaded at runtime and cached in `.fastf1_cache/` (git-ignored). Raw data are **not** committed.
- **Terms:** Formula 1 data belong to their owners. Use for personal / educational analysis; check upstream terms before any redistribution or commercial use.
- **Known data caveats:** lap and sector times come from timing loops; telemetry is ~4-5 Hz with distance integrated from speed;
  `Brake` is boolean; `IsAccurate` and `TrackStatus` are FastF1-derived; corner positions are manually curated (+/- tens of metres).
- **Reproducing a result:** same FastF1 version + same command (stored in `run_metadata.json`). If FastF1 changes its processing, numbers can shift slightly.
