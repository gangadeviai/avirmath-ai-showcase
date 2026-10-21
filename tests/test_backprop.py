"""Independent validation of the backpropagation mathematics."""

import numpy as np
import pytest
import sympy as sp

from avirmath.mathcore import backprop as bp
from avirmath.mathcore.gradient_descent import numerical_gradient


# --- chain rule ----------------------------------------------------------------
@pytest.mark.parametrize("x", [0.3, 1.0, -1.7])
def test_chain_rule_demo_matches_sympy_and_numbers(x):
    t = sp.symbols("t")
    exact = float(sp.diff(sp.sin(t ** 2), t).subs(t, x))
    assert bp.chain_rule_demo(x)[2] == pytest.approx(exact)
    num = numerical_gradient(lambda v: float(np.sin(v[0] ** 2)), [x])[0]
    assert bp.chain_rule_demo(x)[2] == pytest.approx(num, abs=1e-5)


# --- the worked network: every quoted derivative is right ------------------------
def test_worked_network_derivatives_symbolically():
    x, y, w1, w2 = sp.symbols("x y w1 w2", real=True)
    h = 1 / (1 + sp.exp(-w1 * x))
    L = (w2 * h - y) ** 2
    subs = {x: bp.X0, y: bp.Y0, w1: bp.W1_0, w2: bp.W2_0}
    p = bp.forward_backward()
    assert p.dL_dw1 == pytest.approx(float(sp.diff(L, w1).subs(subs)))
    assert p.dL_dw2 == pytest.approx(float(sp.diff(L, w2).subs(subs)))
    assert p.dL_dh == pytest.approx(float(sp.diff((w2 * sp.Symbol("hh") - y) ** 2, sp.Symbol("hh")).subs({w2: bp.W2_0, y: bp.Y0, sp.Symbol("hh"): p.h})))


def test_worked_network_matches_finite_differences():
    p = bp.forward_backward()
    g = numerical_gradient(lambda v: float(bp.loss_surface(v[0], v[1])), [bp.W1_0, bp.W2_0])
    assert np.allclose([p.dL_dw1, p.dL_dw2], g, atol=1e-5)


def test_sigmoid_prime_identity_and_bound():
    z = np.linspace(-6, 6, 200)
    num = numerical_gradient  # noqa: F841 (readability)
    assert np.allclose(bp.sigmoid_prime(z), [numerical_gradient(lambda v: float(bp.sigmoid(v[0])), [zz])[0] for zz in z], atol=1e-5)
    assert bp.sigmoid_prime(z).max() == pytest.approx(bp.SIGMOID_PRIME_MAX, abs=1e-4)


def test_one_backprop_step_lowers_the_loss_and_training_converges():
    traj = bp.train_worked()
    assert traj[1, 2] < traj[0, 2]
    assert np.all(np.diff(traj[:, 2]) <= 1e-12)
    assert traj[-1, 2] < 0.01


# --- general matrix form vs numerical gradients, any depth -----------------------
@pytest.mark.parametrize("sizes", [[3, 4, 2], [5, 6, 6, 3], [2, 3, 3, 3, 1]])
def test_matrix_backprop_equals_numerical_gradient(sizes):
    Ws = bp.init_mlp(sizes, seed=2)
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=sizes[0]), rng.normal(size=sizes[-1])
    grads, _ = bp.mlp_backprop(Ws, x, y)
    num = bp.mlp_numerical_grads(Ws, x, y)
    for g, n in zip(grads, num):
        assert np.allclose(g, n, atol=1e-5)


def test_backward_sweep_reuses_upstream_delta():
    """dL/dW_l = delta_l h_{l-1}^T: the same delta serves every weight into that layer."""
    Ws = bp.init_mlp([3, 4, 2], seed=5)
    x, y = np.array([0.2, -1.0, 0.7]), np.array([1.0, 0.0])
    grads, deltas = bp.mlp_backprop(Ws, x, y)
    _, hs = bp.mlp_forward(Ws, x)
    assert np.allclose(grads[0], np.outer(deltas[0], hs[0]))
    assert np.allclose(grads[1], np.outer(deltas[1], hs[1]))


def test_cost_claim():
    assert bp.nudge_cost(1000) == 1001 and bp.backprop_cost(1000) == 2
    assert bp.nudge_cost(10 ** 9) > 10 ** 9 and bp.backprop_cost(10 ** 9) == 2


# --- vanishing gradients ---------------------------------------------------------
def test_gradients_shrink_towards_the_input_through_sigmoids():
    norms = bp.delta_norms_by_layer()
    assert norms[0] < norms[-1] / 20          # input-side delta far smaller than output-side


# --- industry example --------------------------------------------------------------
def test_digit_network_learns():
    log = bp.train_digits()
    assert log.accuracies[0] < 0.75
    assert log.accuracies[-1] > 0.9
    assert log.losses[-1] < log.losses[0] / 2


def test_lesson_facts_are_consistent():
    f = bp.lesson_facts()
    assert f["loss_after"] < f["loss"]
    assert f["digits_acc_final"] > 90 and f["vanish_ratio"] >= 20
    assert f["w1_new"] == pytest.approx(f["w1"] - f["lr"] * f["dL_dw1"], abs=2e-3)
