# How to turn `report_draft.md` into the final report (aim: 2-3 pages + figures)

1. **Look at every figure first.** For each, write one sentence: "This shows ___, which means ___."
2. **Pick 3-5 observations.** Keep only those where you can state: the number, its uncertainty, one alternative explanation, and what extra data would settle it.
3. **Use calibrated language.**
   - OK: "A is faster by a median 0.31 s (95% CI 0.22-0.40, n=34); the gap persists on clear-air laps."
   - OK: "The pattern is consistent with higher exit speed from T10; brake is on/off only, so technique is not testable."
   - Not OK: "B has a tyre degradation problem." / "A's setup is better." / "The gap is caused by braking."
4. **Answer these four questions explicitly** (they are the brief):
   - Where does the advantage appear: which sector, corner entry or exit?
   - Is it consistent across comparable laps, or driven by one exceptional lap?
   - How does lap time change over a stint, and what else could explain it?
   - Which additional data are needed before concluding tyre / braking / setup causes?
5. **Executive summary last**: question, main finding with uncertainty, strongest alternative explanation, next step.
6. **Sanity checks before submitting**: lap counts per driver >= ~20 clean laps; telemetry laps are the ones you intended; compare 2 telemetry lap pairs (`--tel-laps`); numbers in the text match `tables/*.csv`.
