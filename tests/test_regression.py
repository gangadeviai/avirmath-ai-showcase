"""Independent validation of the linear & logistic regression mathematics."""

import numpy as np
import pytest
import sympy as sp

from avirmath.mathcore import regression as rg
from avirmath.mathcore.gradient_descent import numerical_gradient


def test_closed_form_matches_normal_equations_and_numpy():
    w, b = rg.fit_line(rg.X1, rg.Y1)
    theta = rg.normal_equations(rg.design(rg.X1), rg.Y1)
    assert np.allclose([w, b], theta)
    assert np.allclose([w, b], np.polyfit(rg.X1, rg.Y1, 1))


def test_least_squares_is_the_minimum_of_the_sse():
    w, b = rg.fit_line(rg.X1, rg.Y1)
    gw, gb = rg.sse_grad(w, b, rg.X1, rg.Y1)
    assert abs(gw) < 1e-9 and abs(gb) < 1e-9                       # zero gradient at the optimum
    for dw, db in ((0.1, 0), (-0.1, 0), (0, 0.2), (0.05, -0.1)):
        assert rg.sse(w + dw, b + db, rg.X1, rg.Y1) > rg.sse(w, b, rg.X1, rg.Y1)


def test_sse_gradient_symbolically_and_numerically():
    w, b = sp.symbols("w b")
    S = sum((w * xi + b - yi) ** 2 for xi, yi in zip(rg.X1, rg.Y1))
    gw, gb = rg.sse_grad(0.3, -0.2, rg.X1, rg.Y1)
    assert float(sp.diff(S, w).subs({w: 0.3, b: -0.2})) == pytest.approx(gw)
    assert float(sp.diff(S, b).subs({w: 0.3, b: -0.2})) == pytest.approx(gb)
    num = numerical_gradient(lambda v: rg.sse(v[0], v[1], rg.X1, rg.Y1), [0.3, -0.2])
    assert np.allclose([gw, gb], num, atol=1e-5)


def test_normal_equations_derive_the_closed_form_symbolically():
    """Setting both partial derivatives to zero gives w = Sxy / Sxx and b = ybar - w xbar."""
    w, b = sp.symbols("w b")
    xs = sp.symbols("x1:4"); ys = sp.symbols("y1:4")
    S = sum((w * xi + b - yi) ** 2 for xi, yi in zip(xs, ys))
    sol = sp.solve([sp.diff(S, w), sp.diff(S, b)], [w, b], dict=True)[0]
    xbar, ybar = sum(xs) / 3, sum(ys) / 3
    w_closed = sum((xi - xbar) * (yi - ybar) for xi, yi in zip(xs, ys)) / sum((xi - xbar) ** 2 for xi in xs)
    assert sp.simplify(sol[w] - w_closed) == 0
    assert sp.simplify(sol[b] - (ybar - w_closed * xbar)) == 0


def test_gradient_descent_reaches_the_closed_form():
    w, b = rg.fit_line(rg.X1, rg.Y1)
    traj = rg.gd_line(rg.X1, rg.Y1)
    assert np.allclose(traj[-1, :2], [w, b], atol=1e-3)
    assert np.all(np.diff(traj[:, 2]) <= 1e-9)


def test_plane_fit_recovers_the_generating_coefficients():
    X, y = rg.make_plane_data()
    theta = rg.normal_equations(np.column_stack([X, np.ones(len(y))]), y)
    assert np.allclose(theta, [1.5, -0.8, 2.0], atol=0.15)


def test_sigmoid_and_logit_are_inverses_and_logit_is_log_odds():
    for p in (0.1, 0.5, 0.9):
        assert rg.sigmoid(rg.logit(p)) == pytest.approx(p)
        assert rg.logit(p) == pytest.approx(np.log(p / (1 - p)))
    assert rg.sigmoid(0) == 0.5


def test_logistic_gradient_is_prediction_minus_truth():
    X, y = rg.HOURS.reshape(-1, 1), rg.PASSED
    w, b = np.array([0.4]), -1.0
    gw, gb = rg.nll_grad(w, b, X, y)
    num = numerical_gradient(lambda v: rg.nll(v[:1], v[1], X, y), [0.4, -1.0])
    assert np.allclose([gw[0], gb], num, atol=1e-5)
    p = rg.sigmoid(X @ w + b)
    assert gw[0] == pytest.approx(float(np.sum((p - y) * X[:, 0])))


def test_logistic_fit_separates_the_hours_example():
    w, b, hist = rg.fit_logistic(rg.HOURS.reshape(-1, 1), rg.PASSED, lr=0.1, steps=2000)
    assert w[0] > 0 and hist[-1][2] < hist[0][2]
    boundary = -b / w[0]
    assert 2.0 < boundary < 3.5
    assert rg.sigmoid(w[0] * 5 + b) > 0.9 and rg.sigmoid(w[0] * 0.5 + b) < 0.1


def test_credit_model_learns_and_signs_make_sense():
    Xs, y, raw = rg.make_credit_data()
    w, b, hist = rg.fit_logistic(Xs, y, lr=0.002, steps=400)
    assert rg.accuracy(w, b, Xs, y) > 0.70
    assert w[0] < 0 < w[1] and w[2] > 0                              # income lowers risk; debt and late payments raise it
    assert hist[-1][2] < hist[0][2]
    assert rg.odds_ratio(0.0) == 1.0


def test_lesson_facts():
    f = rg.lesson_facts()
    assert f["w"] == pytest.approx(1.05, abs=0.01) and f["sse_fit"] < f["sse_zero"]
    assert f["log_p_4h"] > 0.5 > f["log_p_2h"] and f["credit_acc"] > 70


# ---------------------------------------------------------------------------
# Real datasets (lesson v3): German Credit and hourly bike rentals
# ---------------------------------------------------------------------------
from avirmath.mathcore import regression as _rg


def test_german_credit_loads_and_the_model_beats_the_approve_everyone_baseline():
    X, y = _rg.load_german()
    assert X.shape == (1000, 8) and abs(y.mean() - 0.30) < 1e-9
    c = _rg.credit_model()
    acc = np.mean((c["p_te"] >= 0.5) == (c["yte"] > 0.5))
    assert acc > 1 - c["ytr"].mean() and 0.70 < _rg.auc(c["p_te"], c["yte"]) < 0.90
    assert c["hist"][-1][2] < c["hist"][0][2] and _rg.ece(c["p_te"], c["yte"]) < 0.12


def test_auc_and_ece_behave_like_the_definitions():
    assert _rg.auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0 and _rg.auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert abs(_rg.auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) - 0.5) < 1e-12
    rng = np.random.default_rng(0); p = rng.uniform(0, 1, 20000); y = (rng.random(20000) < p).astype(float)
    assert _rg.ece(p, y) < 0.02


def test_threshold_trades_flagged_loans_for_recall():
    c = _rg.credit_model()
    a, b = _rg.confusion(c["p_te"], c["yte"], 0.5), _rg.confusion(c["p_te"], c["yte"], 0.3)
    assert b["recall_bad"] > a["recall_bad"] and b["flagged"] > a["flagged"]


def test_bike_rentals_linear_model_and_its_failure_modes():
    b = _rg.bike_model()
    assert b["n"] == 17379 and b["r2"] > 0.3 and b["rmse"] < b["ytr"].std()
    assert b["leak_rmse"] < 1e-6                       # casual + registered == cnt: leakage gives a perfect fit
    assert b["extrap_pred"] > b["in_range_pred"] and 60.0 > b["t_max"]
    assert b["resid_by_hour"][17] > 0 and b["resid_by_hour"][3] < 0   # rush hour under-predicted, night over-predicted
