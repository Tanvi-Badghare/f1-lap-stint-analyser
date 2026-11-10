import numpy as np
from f1analyser import stats as st


def test_bootstrap_ci_contains_true_median():
    rng = np.random.default_rng(0)
    x = rng.normal(0.3, 0.1, 60)
    est, lo, hi = st.bootstrap_ci(x, np.median, 2000)
    assert lo < 0.3 < hi and lo < est < hi


def test_bootstrap_small_sample_is_nan():
    assert np.isnan(st.bootstrap_ci([1.0, 2.0])[1])


def test_linear_trend_recovers_slope():
    x = np.arange(20)
    y = 90 + 0.05 * x + np.random.default_rng(1).normal(0, 0.02, 20)
    r = st.linear_trend(x, y)
    assert abs(r["slope"] - 0.05) < 0.005 and r["ci_low"] < 0.05 < r["ci_high"]


def test_sign_share():
    share, n, p = st.sign_share([0.1] * 18 + [-0.1] * 2)
    assert n == 20 and abs(share - 0.9) < 1e-9 and p < 0.01


def test_mad():
    assert abs(st.mad([1, 2, 3, 4, 100]) - 1.4826 * 1) < 1e-9
