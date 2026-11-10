"""Build a website data bundle.

  python -m f1analyser.build_web --year 2026 --gp Japan --drivers AAA BBB              (OpenF1, default)
  python -m f1analyser.build_web --year 2026 --gp Japan --screen                       (list teammate pairs first)
  python -m f1analyser.build_web --source fastf1 --year 2025 --gp Japan --drivers A B  (FastF1)
"""
from __future__ import annotations

import argparse

from .config import AnalysisConfig
from .service import compute_all, save_bundle


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="f1analyser.build_web")
    p.add_argument("--source", choices=["openf1", "fastf1"], default="openf1",
                   help="openf1 (default, free, includes 2026) or fastf1")
    p.add_argument("--year", type=int)
    p.add_argument("--gp", help="event name (e.g. Japan) or round number")
    p.add_argument("--session", default="R")
    p.add_argument("--drivers", nargs=2, metavar=("A", "B"))
    p.add_argument("--tel-laps", nargs=2, type=int, metavar=("LAP_A", "LAP_B"))
    p.add_argument("--cache-dir", default=".f1cache")
    p.add_argument("--out-dir", default="web_data")
    p.add_argument("--no-telemetry", action="store_true", help="skip telemetry (laps, sectors, stints only)")
    p.add_argument("--allow-yellow", action="store_true")
    p.add_argument("--screen", action="store_true", help="only list clean-lap counts and teammate pairs")
    p.add_argument("--demo", action="store_true", help="write the SYNTHETIC demo bundle instead")
    a = p.parse_args(argv)
    if a.demo:
        from .demo import demo_result
        res = demo_result()
    else:
        if not (a.year and a.gp):
            p.error("--year and --gp are required (or use --demo)")
        if a.screen:
            if a.source == "openf1":
                from .openf1 import screen_openf1
                screen_openf1(a.year, a.gp, a.session, a.cache_dir)
            else:
                from .pipeline import screen_session
                screen_session(a.year, a.gp, a.session, a.cache_dir)
            return 0
        if not a.drivers:
            p.error("--drivers A B is required (run with --screen first to choose a pair)")
        cfg = AnalysisConfig(year=a.year, gp=a.gp, session=a.session, driver_a=a.drivers[0].upper(),
                             driver_b=a.drivers[1].upper(), cache_dir=a.cache_dir, allow_yellow=a.allow_yellow, use_telemetry=not a.no_telemetry,
                             tel_laps=tuple(a.tel_laps) if a.tel_laps else None)
        if a.source == "openf1":
            from .openf1 import compute_openf1
            res = compute_openf1(cfg)
        else:
            res = compute_all(cfg)
    out = save_bundle(res, a.out_dir)
    print(f"Saved website data to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
