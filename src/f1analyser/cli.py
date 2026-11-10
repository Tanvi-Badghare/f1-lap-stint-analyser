"""Command line: `python -m f1analyser screen|analyse ...`"""
from __future__ import annotations

import argparse
import shlex
import sys

from .config import AnalysisConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="f1analyser", description="F1 Lap & Stint Performance Analyser")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--year", type=int, required=True)
        sp.add_argument("--gp", required=True, help="event name (e.g. Bahrain) or round number")
        sp.add_argument("--session", default="R", help="R (default), Q, S, FP1...")
        sp.add_argument("--cache-dir", default=".fastf1_cache")
        sp.add_argument("--allow-yellow", action="store_true")

    sc = sub.add_parser("screen", help="check clean-lap counts and teammate pairs before choosing")
    common(sc)

    an = sub.add_parser("analyse", help="run the full analysis for two drivers")
    common(an)
    an.add_argument("--drivers", nargs=2, required=True, metavar=("A", "B"), help="3-letter codes, e.g. VER PER")
    an.add_argument("--output-dir", default="outputs")
    an.add_argument("--tel-mode", choices=["auto", "fastest", "matched"], default="auto")
    an.add_argument("--tel-laps", nargs=2, type=int, metavar=("LAP_A", "LAP_B"), help="override telemetry lap selection")
    an.add_argument("--traffic-gap", type=float, default=1.5)
    an.add_argument("--no-is-accurate", action="store_true")
    an.add_argument("--any-compound", action="store_true", help="pair laps even if compounds differ")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from .pipeline import run_analysis, screen_session

    if args.cmd == "screen":
        screen_session(args.year, args.gp, args.session, args.cache_dir,
                       AnalysisConfig(year=args.year, gp=args.gp, session=args.session, allow_yellow=args.allow_yellow))
        return 0
    cfg = AnalysisConfig(
        year=args.year, gp=args.gp, session=args.session, driver_a=args.drivers[0].upper(),
        driver_b=args.drivers[1].upper(), cache_dir=args.cache_dir, output_dir=args.output_dir,
        allow_yellow=args.allow_yellow, use_is_accurate=not args.no_is_accurate, traffic_gap_s=args.traffic_gap,
        tel_mode=args.tel_mode, tel_laps=tuple(args.tel_laps) if args.tel_laps else None,
        paired_same_compound=not args.any_compound)
    run_analysis(cfg, command="python -m f1analyser " + " ".join(shlex.quote(x) for x in (argv or sys.argv[1:])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
