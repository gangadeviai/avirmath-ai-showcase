"""Independent validation of the LMS noise canceller and the rugged landscape."""

import numpy as np
import pytest

from avirmath.mathcore import adaptive_filter as af
from avirmath.mathcore import gradient_descent as gd


def test_lms_discovers_the_unknown_leak():
    run = af.run_lms()
    assert np.allclose(run.weights[-500:].mean(axis=0), af.LEAK, atol=0.05)


def test_lms_cleans_the_signal_by_a_large_margin():
    run = af.run_lms()
    before = af.snr_db(run.signal, run.primary)
    after = af.snr_db(run.signal[-1000:], run.output[-1000:])
    assert before < 0            # noise starts out louder than the signal
    assert after - before > 15   # at least 15 dB better


def test_error_surface_formula_matches_measured_error():
    rng = np.random.default_rng(0)
    n = 200_000
    s = af.wanted_signal(n)
    x = rng.normal(size=n)
    xp = np.concatenate([[0.0], x[:-1]])
    d = s + af.LEAK[0] * x + af.LEAK[1] * xp
    for w in ([0.0, 0.0], [0.3, 0.4], [0.8, -0.5]):
        measured = np.mean((d - w[0] * x - w[1] * xp) ** 2)
        assert measured == pytest.approx(af.error_surface(w[0], w[1], np.mean(s ** 2)), rel=0.02)


def test_lms_update_is_a_one_sample_gradient_step():
    """d(e^2)/dw = -2 e u, so w + 2 mu e u is exactly w - mu * grad(e^2)."""
    w, u, d, mu = np.array([0.2, -0.1]), np.array([0.7, -1.3]), 0.9, 0.01
    e = d - w @ u
    grad = gd.numerical_gradient(lambda v: float((d - v @ u) ** 2), w)
    assert np.allclose(w + 2 * mu * e * u, w - mu * grad, atol=1e-8)


def test_rugged_gradient_and_descent():
    for p in ([2.5, 2.1], [-1.0, 0.4], [0.3, -2.2]):
        num = gd.numerical_gradient(lambda v: float(af.rugged(v[0], v[1])), p)
        assert np.allclose(af.rugged_grad(p), num, atol=1e-5)
    path = af.rugged_path()
    losses = af.rugged(path[:, 0], path[:, 1])
    assert np.all(np.diff(losses) < 1e-9)                      # never climbs
    assert np.linalg.norm(af.rugged_grad(path[-1])) < 0.05     # settles in a valley
