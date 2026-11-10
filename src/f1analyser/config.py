"""Central configuration. Every analysis choice that changes a number lives here."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class AnalysisConfig:
    year: int
    gp: str                      # event name ("Bahrain") or round number ("1")
    session: str = "R"           # R, Q, S, SQ, FP1.. (stint analysis needs R or a long run)
    driver_a: str = "VER"        # reference driver (3-letter code)
    driver_b: str = "PER"        # comparison driver
    cache_dir: str = ".fastf1_cache"
    output_dir: str = "outputs"

    # --- lap cleaning -------------------------------------------------
    allow_yellow: bool = False   # False: only track status "1" (green) laps are kept
    use_is_accurate: bool = True # respect FastF1's IsAccurate flag
    quick_lap_threshold: float = 1.07  # drop laps slower than 107% of the driver's best valid lap
    traffic_gap_s: float = 1.5   # gap-to-car-ahead (at timing line) below which a lap is "in traffic"

    # --- statistics ---------------------------------------------------
    n_bootstrap: int = 5000
    seed: int = 42
    paired_same_compound: bool = True

    # --- stints -------------------------------------------------------
    min_stint_laps: int = 6
    # Assumed fuel-burn lap-time benefit (s per lap). NOT measured here; used only as a sensitivity range.
    fuel_effects: tuple = (0.0, 0.02, 0.03, 0.04)
    min_cars_for_field_ref: int = 5

    # --- telemetry ----------------------------------------------------
    tel_mode: str = "auto"       # auto | fastest | matched
    tel_laps: tuple | None = None  # explicit lap numbers (lap_a, lap_b)
    use_telemetry: bool = True   # False = laps/sectors/stints only (also used automatically if the source refuses telemetry)
    grid_step_m: float = 5.0
    n_minisectors: int = 25
    corner_before_m: float = 150.0
    corner_after_m: float = 150.0
    full_throttle_pct: float = 95.0

    def slug(self) -> str:
        gp = str(self.gp).lower().replace(" ", "-")
        return f"{self.year}_{gp}_{self.session}_{self.driver_a}-vs-{self.driver_b}"

    def to_dict(self) -> dict:
        return asdict(self)
