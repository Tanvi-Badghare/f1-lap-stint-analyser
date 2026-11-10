"""Print the aligned telemetry around a distance window.  python inspect_chicane.py web_data\\2026_japan_R_PIA-vs-NOR 5200 5500"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

d, lo, hi = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
a = pd.read_csv(d / "aligned.csv")
w = a[(a.Distance >= lo) & (a.Distance <= hi)].copy()
w["dSpeed_B-A"] = w.Speed_B - w.Speed_A
# slope check: how much delta SHOULD change per 5 m from the two speed traces (1/vB - 1/vA)
w["expected_dDelta"] = 5.0 * (3.6 / w.Speed_B.clip(lower=1) - 3.6 / w.Speed_A.clip(lower=1))
w["actual_dDelta"] = w.Delta.diff().shift(-1)
cols = ["Distance", "Speed_A", "Speed_B", "dSpeed_B-A", "Brake_A", "Brake_B", "Delta", "expected_dDelta", "actual_dDelta"]
pd.set_option("display.width", 200)
print(w[cols].iloc[::2].round(3).to_string(index=False))
ok = (w.expected_dDelta - w.actual_dDelta).abs().median()
print(f"\nmedian |expected - actual| change of delta per 5 m: {ok:.4f} s")
print("small -> the jump is what the two speed traces say (a real speed difference at the same distance);")
print("large -> the delta curve moves for another reason (distance matching), i.e. an alignment artifact.")