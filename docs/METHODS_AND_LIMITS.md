# Methods and limits

## Sign convention
`delta = time_B - time_A`. Positive: A faster. Telemetry `Delta(d) = t_B(d) - t_A(d)`; a rising curve means A gains.

## Confounders and what controls them (partially)
| Confounder | Control used | What remains |
|---|---|---|
| Traffic / dirty air | lap-end gap proxy (<1.5 s); clear-air repeat of key numbers | gaps *during* the lap are unknown |
| Fuel load | pair by lap number; fuel-assumption sensitivity (0-0.04 s/lap, an assumption) | actual fuel mass unknown; teams fuel differently |
| Track evolution / wind / temperature | pace relative to field median on the same lap | field's own compounds and ages differ |
| Tyre state | same-compound pairing; tyre age as regressor | temperature, pressure, set history unknown |
| Driver management | slow-lap filter (107 %) | lift-and-coast, engine modes, gap management are invisible |
| Outlier laps | bootstrap CI, leave-one-out, top-5 vs best vs median | |

## Why "apparent trend", not degradation
Within a stint, tyre age, fuel burn and track evolution all advance together lap by lap. Regression on tyre age
cannot separate them; the fuel adjustment is an assumed value, and the field-relative slope only removes effects shared by the field.

## Telemetry caveats
Distance is integrated speed, so two laps have slightly different lengths; B is rescaled to A's length (factor reported).
Brake points are accurate to roughly one sample (15-20 m at high speed). Brake pressure and steering are not available, so
"braking technique" can only be described as brake-on distance, not inferred as technique.

## What would be needed for causal claims
Tyre degradation: fuel mass, tyre temps/pressures, long runs on the same compound, several drivers and stints.
Setup: ride height, wing angles, downforce/drag from sims or wind-tunnel, tyre pressures, driver feedback. Without these,
write "consistent with" and name the extra data.
