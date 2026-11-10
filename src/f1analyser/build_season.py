"""Build bundles for every race round of a season from OpenF1 (teammate pair chosen automatically).

  python -m f1analyser.build_season --year 2026 --rounds 1 2 3
Skips failures and writes a log line per round. Pair = the two drivers of the team with the most clean laps.
"""
from __future__ import annotations

import argparse
import traceback

from .config import AnalysisConfig
from .service import save_bundle


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="f1analyser.build_season")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--rounds", nargs="+", type=int, required=True)
    p.add_argument("--session", default="R")
    p.add_argument("--cache-dir", default=".f1cache")
    p.add_argument("--out-dir", default="web_data")
    a = p.parse_args(argv)
    from .openf1 import compute_openf1, screen_openf1
    ok, bad = [], []
    for rnd in a.rounds:
        try:
            scr = screen_openf1(a.year, rnd, a.session, a.cache_dir)
            pair = _pick_pair(scr)
            if pair is None:
                raise RuntimeError("no teammate pair with enough clean laps")
            cfg = AnalysisConfig(year=a.year, gp=rnd, session=a.session, driver_a=pair[0],
                                 driver_b=pair[1], cache_dir=a.cache_dir)
            save_bundle(compute_openf1(cfg), a.out_dir)
            ok.append((rnd, pair))
            print(f"round {rnd}: saved {pair[0]} vs {pair[1]}")
        except Exception as e:  # keep going; one bad round must not stop the season
            bad.append(rnd)
            print(f"round {rnd}: FAILED ({type(e).__name__}: {e})")
            traceback.print_exc(limit=1)
    print(f"done: {len(ok)} saved, {len(bad)} failed {bad}")
    return 0 if not bad else 1


def _pick_pair(pairs):
    """pairs = DataFrame from screen_openf1 (best first); take the top pair with enough clean laps."""
    if pairs is None or len(pairs) == 0:
        return None
    r = pairs.iloc[0]
    return (r["driver_a"], r["driver_b"]) if int(r["min_clean"]) >= 8 else None


if __name__ == "__main__":
    raise SystemExit(main())
