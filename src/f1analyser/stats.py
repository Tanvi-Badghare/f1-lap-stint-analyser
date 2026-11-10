"""Small, dependency-light statistics helpers (all unit-tested on synthetic data)."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def mad(x) -> float:
    """Median absolute deviation, scaled to be comparable with a standard deviation."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def bootstrap_ci(values, stat=np.median, n_boot: int = 5000, alpha: float = 0.05, seed: int = 42):
    """Percentile bootstrap CI of ``stat(values)``. Returns (estimate, low, high)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 3:
        return (float(stat(v)) if v.size else float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = stat(v[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(v)), float(lo), float(hi)


def bootstrap_diff_ci(x, y, stat=np.median, n_boot: int = 5000, alpha: float = 0.05, seed: int = 42):
    """Bootstrap CI of ``stat(y) - stat(x)`` for two *unpaired* samples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if x.size < 3 or y.size < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bx = stat(x[rng.integers(0, x.size, size=(n_boot, x.size))], axis=1)
    by = stat(y[rng.integers(0, y.size, size=(n_boot, y.size))], axis=1)
    d = by - bx
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(y) - stat(x)), float(lo), float(hi)


def sign_share(deltas):
    """Share of strictly positive deltas, n, and two-sided binomial p-value vs 50/50."""
    d = np.asarray(deltas, dtype=float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    n = int(d.size)
    if n == 0:
        return float("nan"), 0, float("nan")
    k = int((d > 0).sum())
    p = float(sps.binomtest(k, n, 0.5).pvalue)
    return k / n, n, p


def linear_trend(x, y, alpha: float = 0.05) -> dict:
    """OLS slope of y on x with standard error and a t-based CI."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = int(x.size)
    nan = float("nan")
    if n < 3 or np.ptp(x) == 0:
        return dict(n=n, slope=nan, stderr=nan, ci_low=nan, ci_high=nan, intercept=nan, r2=nan,
                    resid_std=nan, p=nan)
    res = sps.linregress(x, y)
    t = sps.t.ppf(1 - alpha / 2, n - 2)
    resid = y - (res.intercept + res.slope * x)
    return dict(
        n=n,
        slope=float(res.slope),
        stderr=float(res.stderr),
        ci_low=float(res.slope - t * res.stderr),
        ci_high=float(res.slope + t * res.stderr),
        intercept=float(res.intercept),
        r2=float(res.rvalue ** 2),
        resid_std=float(resid.std(ddof=2)),
        p=float(res.pvalue),
    )
