# Data source: OpenF1

**Why:** since about 15 October 2026 F1's timing archive answers HTTP 401 (F1 TV login required), so FastF1 cannot load
sessions that are not already in its cache. OpenF1 (https://openf1.org) serves historical data from 2023 onwards for free
without authentication, including 2026.

## Commands
```powershell
python -m f1analyser.build_web --year 2026 --gp Japan --screen
python -m f1analyser.build_web --year 2026 --gp Japan --drivers AAA BBB
```
`--gp` accepts a name (`Japan`, `Suzuka`, `Japanese Grand Prix`) or the race round number (testing excluded).
Responses are cached in `.f1cache/openf1/`; a run needs about 10 requests, and a retry continues from the cache.

## How OpenF1 data is mapped
| Analysis needs | OpenF1 source | Note |
|---|---|---|
| Lap and sector times | `laps` | seconds |
| Compound, stint, tyre age | `stints` | tyre age = `tyre_age_at_start` + laps into stint |
| In/out laps | `stints` + `laps.is_pit_out_lap` | |
| Track status (SC, VSC, red, yellow) | `race_control` messages | derived windows; **sector-only yellows are not detected** |
| Deleted laps | `race_control` "DELETED" messages | |
| `IsAccurate` | lap time and all three sectors present | proxy |
| Lap end times (traffic proxy) | next lap's `date_start` | `date_start` is documented as approximate |
| Telemetry | `car_data` + `location` (about 3.7 Hz) | brake is 0/100 and is read as on/off |
| Corners | MultiViewer file linked in `meetings.circuit_info_url` | used only if coordinates match the car positions; otherwise the corner table is skipped |

## Extra caveats compared with FastF1
- Lap `date_start` is approximate, so each telemetry lap is re-anchored on the same track line using the car position,
  and its duration is the official lap time. A residual error of a few metres at the start/finish line is possible.
- Distance is integrated from integer km/h speed; B's distance is rescaled to A's length (factor shown on the Telemetry tab).
- Track-status and lap-accuracy flags are derived, not official, so check the exclusion table before trusting small samples.
- Check OpenF1's terms before publishing their data widely. Credit OpenF1 as the source.
