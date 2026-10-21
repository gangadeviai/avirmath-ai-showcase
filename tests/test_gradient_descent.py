"""Independent validation of the gradient descent math.

If any of these fail, nothing downstream (lesson page, experiment, video) may be built.
"""

import numpy as np
import pytest
import sympy as sp

from avirmath.mathcore import gradient_descent as gd


# --- analytic gradients agree with finite differences -----------------------
@pytest.mark.parametrize("s", [-3.0, 0.0, 2.0, 8.0, 11.5])
def test_chai_gradient_matches_numerical(s):
    num = gd.numerical_gradient(lambda x: float(gd.chai_loss(x[0])), [s])[0]
    assert gd.chai_grad(s) == pytest.approx(num, abs=1e-5)


@pytest.mark.parametrize("p", [(2.0, 5.0), (8.0, 3.0), (-1.0, 0.5), (10.0, 10.0)])
def test_bowl_gradient_matches_numerical(p):
    bowl = gd.Bowl()
    assert np.allclose(bowl.grad(p), gd.numerical_gradient(bowl.loss, p), atol=1e-5)


def test_mse_gradient_matches_numerical():
    X, y = gd.make_eta_dataset(n=80)
    Xs, _, _ = gd.standardize(X)
    rng = np.random.default_rng(1)
    w, b = rng.normal(size=3), 1.7
    gw, gb = gd.mse_grad(w, b, Xs, y)
    f = lambda v: gd.mse_loss(v[:3], v[3], Xs, y)
    num = gd.numerical_gradient(f, np.append(w, b))
    assert np.allclose(np.append(gw, gb), num, atol=1e-4)


# --- the formulas shown on screen are the formulas in the code ---------------
def test_chai_formula_symbolically():
    s, s_best = sp.symbols("s s_best", real=True)
    shown_loss = (s - s_best) ** 2
    shown_grad = 2 * (s - s_best)
    assert sp.simplify(sp.diff(shown_loss, s) - shown_grad) == 0
    # and the code computes exactly that expression
    f = sp.lambdify((s, s_best), shown_grad)
    for val in (-2.0, 3.0, 9.5):
        assert gd.chai_grad(val, 8.0) == pytest.approx(f(val, 8.0))


def test_error_ratio_derivation_symbolically():
    s, s_best, lr = sp.symbols("s s_best lr", real=True)
    new_s = s - lr * 2 * (s - s_best)
    assert sp.simplify((new_s - s_best) - (1 - 2 * lr) * (s - s_best)) == 0


def test_mse_gradient_formula_symbolically():
    w, b = sp.symbols("w b", real=True)
    xs, ys = [1.5, -0.5, 2.0], [4.0, 1.0, 6.5]
    L = sum((w * x + b - y) ** 2 for x, y in zip(xs, ys)) / 3
    X, y = np.array(xs).reshape(-1, 1), np.array(ys)
    gw, gb = gd.mse_grad(np.array([0.7]), -0.3, X, y)
    assert float(sp.diff(L, w).subs({w: 0.7, b: -0.3})) == pytest.approx(gw[0])
    assert float(sp.diff(L, b).subs({w: 0.7, b: -0.3})) == pytest.approx(gb)


# --- learning-rate behaviour: the claims the lesson makes ---------------------
def test_chai_error_shrinks_by_exact_ratio():
    path = gd.chai_path(lr=0.1, steps=10)
    err = path - gd.SUGAR_BEST
    assert np.allclose(err[1:] / err[:-1], gd.chai_error_ratio(0.1))
    assert gd.chai_error_ratio(0.1) == pytest.approx(0.8)


def test_chai_lr_half_lands_in_one_step():
    assert gd.chai_path(lr=0.5, steps=1)[1] == pytest.approx(gd.SUGAR_BEST)


def test_chai_lr_one_bounces_forever():
    path = gd.chai_path(lr=1.0, steps=6)
    assert np.allclose(path, [2, 14, 2, 14, 2, 14, 2])


def test_chai_lr_above_limit_diverges():
    err = np.abs(gd.chai_path(lr=1.1, steps=10) - gd.SUGAR_BEST)
    assert np.all(np.diff(err) > 0)


@pytest.mark.parametrize("lr, regime", [
    (0.1, "converges smoothly"), (0.5, "one step"), (0.9, "converges with overshoot"),
    (1.0, "oscillates forever"), (1.1, "diverges"), (0.0, "stuck"),
])
def test_chai_regimes(lr, regime):
    assert gd.chai_regime(lr) == regime


def test_bowl_stability_limit_is_set_by_steepest_direction():
    bowl = gd.Bowl()
    assert bowl.lr_max == pytest.approx(0.125)
    assert bowl.condition_number == pytest.approx(8.0)
    ok = bowl.path(lr=bowl.lr_max * 0.96, steps=400)
    bad = bowl.path(lr=bowl.lr_max * 1.04, steps=400)
    assert bowl.loss(ok[-1]) < 1e-6
    assert bowl.loss(bad[-1]) > bowl.loss(bad[0])


def test_bowl_zigzags_across_the_steep_direction():
    """Near the limit the tea-strength coordinate flips sign every step: the zig-zag."""
    path = gd.Bowl().path(lr=0.11, steps=8)
    t_err = path[:, 1] - 3.0
    assert np.all(t_err[1:] * t_err[:-1] < 0)


# --- industry example ------------------------------------------------------
def test_full_batch_gd_reaches_least_squares_solution():
    X, y = gd.make_eta_dataset()
    Xs, _, _ = gd.standardize(X)
    log = gd.train_eta(Xs, y, lr=0.1, epochs=400, batch_size=len(y))
    w_star, b_star = gd.least_squares(Xs, y)
    assert np.allclose(log.weights[-1], w_star, atol=1e-6)
    assert log.biases[-1] == pytest.approx(b_star, abs=1e-6)


def test_sgd_training_gets_close_to_the_noise_floor():
    X, y = gd.make_eta_dataset()
    Xs, _, _ = gd.standardize(X)
    log = gd.train_eta(Xs, y)
    noise_floor = gd.NOISE_MIN * np.sqrt(2 / np.pi)          # E|N(0, sigma)|
    assert log.maes[0] > 20                                   # untrained: wildly wrong
    assert log.maes[-1] < noise_floor * 1.15                  # trained: about as good as possible
    assert log.losses[-1] < log.losses[0] / 50


def test_learned_weights_recover_the_real_world_meaning():
    """Un-standardised, the model should rediscover ~3.2 min/km etc."""
    X, y = gd.make_eta_dataset()
    Xs, mu, sd = gd.standardize(X)
    log = gd.train_eta(Xs, y, lr=0.1, epochs=400, batch_size=len(y))
    w_raw = log.weights[-1] / sd
    b_raw = log.biases[-1] - np.sum(w_raw * mu)
    assert np.allclose(w_raw, gd.TRUE_WEIGHTS, rtol=0.1)
    assert b_raw == pytest.approx(gd.TRUE_BIAS, abs=1.5)


def test_standardising_features_rounds_the_bowl():
    X, _ = gd.make_eta_dataset()
    Xs, _, _ = gd.standardize(X)
    assert gd.mse_condition_number(X) > 100 * gd.mse_condition_number(Xs)
    assert gd.mse_condition_number(Xs) < 2


def test_mse_stability_limit():
    X, y = gd.make_eta_dataset()
    Xs, _, _ = gd.standardize(X)
    limit = gd.mse_lr_max(Xs)
    ok = gd.train_eta(Xs, y, lr=limit * 0.95, epochs=300, batch_size=len(y))
    bad = gd.train_eta(Xs, y, lr=limit * 1.05, epochs=300, batch_size=len(y))
    assert ok.losses[-1] < ok.losses[0]
    assert bad.losses[-1] > bad.losses[0]


# --- learning-rate schedule --------------------------------------------------
def test_warmup_cosine_shape():
    steps = np.arange(0, 1001)
    lr = gd.warmup_cosine(steps, total_steps=1000, peak_lr=3e-4, warmup_steps=100)
    assert lr[0] == 0
    assert lr[100] == pytest.approx(3e-4)
    assert lr[-1] == pytest.approx(3e-5)
    assert np.all(np.diff(lr[:101]) > 0)
    assert np.all(np.diff(lr[100:]) <= 1e-18)


# --- facts the narrator is allowed to say ------------------------------------
def test_lesson_facts_are_the_story_we_tell():
    f = gd.lesson_facts()
    assert f["chai_grad_start"] == -12
    assert (f["chai_day1"], f["chai_day2"], f["chai_day3"]) == (3.2, 4.16, 4.93)
    assert f["chai_shrink_percent"] == 20
    assert f["chai_bouncy_day1"] > f["sugar_best"] > f["sugar_start"]      # genuinely overshoots
    assert f["eta_mae_start"] > 30 and f["eta_mae_final"] < 3
    assert f["eta_mae_start"] > f["eta_mae_epoch1"] > f["eta_mae_final"]
    assert f["eta_min_per_km_learned"] == pytest.approx(f["eta_min_per_km_true"], abs=0.15)
    assert f["eta_curvature_ratio_raw"] > 1000 and f["eta_curvature_ratio_scaled"] < 1.5


# --- the opening story is a real run of noisy (stochastic) gradient descent ----
def test_story_path_matches_what_the_narrator_says():
    path, felt = gd.chai_noisy_path()
    assert path[0] == gd.SUGAR_START
    assert np.all(np.diff(path[:4]) > 0)            # "adds a little... a bit more"
    assert np.all(felt[:4] < 0)                     # complaints: not sweet enough
    assert path[4] > gd.SUGAR_BEST and felt[4] > 0   # "one day they flip: too sweet!"
    assert path[5] < path[4]                        # "and she steps back"
    assert abs(path[5] - gd.SUGAR_BEST) < 0.25      # ...landing close to the sweet spot


def test_noisy_feedback_still_finds_the_sweet_spot_on_average():
    ends = [gd.chai_noisy_path(steps=40, seed=s)[0][-10:].mean() for s in range(60)]
    assert np.mean(ends) == pytest.approx(gd.SUGAR_BEST, abs=0.15)


def test_noisy_two_knob_path_wanders_but_arrives():
    bowl = gd.Bowl()
    smooth = bowl.path(lr=0.04, steps=60)
    noisy = gd.bowl_noisy_path(bowl, lr=0.04, steps=60)
    assert bowl.loss(smooth[-1]) < 0.01
    assert np.linalg.norm(noisy[-15:].mean(axis=0) - bowl.best) < 0.6      # ends in the right neighbourhood
    assert np.abs(np.diff(noisy, axis=0)).sum() > np.abs(np.diff(smooth, axis=0)).sum()   # by a longer road


def test_landscape_walk_is_steepest_descent_and_reaches_the_valley():
    path = gd.landscape_path()
    g0 = gd.LANDSCAPE.grad(path[0])
    assert np.allclose(path[1] - path[0], -gd.LANDSCAPE_LR * g0)          # each step is exactly -eta * gradient
    tangent_to_contour = np.array([-g0[1], g0[0]])
    assert abs(np.dot(path[1] - path[0], tangent_to_contour)) < 1e-12     # so it crosses contours at right angles
    losses = [gd.LANDSCAPE.loss(p) for p in path]
    assert np.all(np.diff(losses) < 0) and losses[-1] < 0.05
