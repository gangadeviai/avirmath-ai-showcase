"""Independent validation of the softmax & cross-entropy mathematics."""

import numpy as np
import pytest
import sympy as sp

from avirmath.mathcore import softmax as sm


def test_softmax_is_a_distribution_and_matches_textbook_formula():
    p = sm.softmax(sm.SCORES)
    assert p.sum() == pytest.approx(1.0) and np.all(p > 0)
    assert np.allclose(p, sm.softmax_naive(sm.SCORES))
    assert np.allclose(p, np.exp(sm.SCORES) / np.exp(sm.SCORES).sum())


def test_worked_numbers():
    p = sm.softmax([2.0, 1.0, 0.1])
    assert np.allclose(np.round(p, 3), [0.659, 0.242, 0.099])
    assert np.argmax(p) == 0


def test_shift_invariance_and_overflow_protection():
    assert np.allclose(sm.softmax(sm.SCORES), sm.softmax(sm.SCORES + 100))
    big = np.array([1000.0, 999.0, 0.0])
    assert np.isfinite(sm.softmax(big)).all()
    with np.errstate(over="ignore", invalid="ignore"):
        assert not np.isfinite(sm.softmax_naive(big)).all()


def test_temperature_limits():
    cold, hot = sm.softmax(sm.SCORES, 0.01), sm.softmax(sm.SCORES, 1000)
    assert cold[0] > 0.999                              # winner takes all
    assert np.allclose(hot, 1 / 3, atol=1e-3)           # uniform
    assert sm.softmax(sm.SCORES, 0.3)[0] > sm.softmax(sm.SCORES, 1)[0] > sm.softmax(sm.SCORES, 5)[0]


def test_two_classes_reduce_to_sigmoid():
    a, b = sm.two_class_softmax_equals_sigmoid(1.7, -0.4)
    assert a == pytest.approx(b)


def test_boltzmann_is_softmax_of_minus_energy():
    E = np.array([0.0, 1.0, 2.0])
    p = sm.boltzmann(E, kT=0.5)
    assert np.allclose(p, np.exp(-E / 0.5) / np.exp(-E / 0.5).sum())
    assert p[0] > p[1] > p[2]


def test_softmax_jacobian_symbolically_and_numerically():
    z1, z2, z3 = sp.symbols("z1 z2 z3")
    Z = sp.exp(z1) + sp.exp(z2) + sp.exp(z3)
    p1 = sp.exp(z1) / Z
    d11 = sp.simplify(sp.diff(p1, z1) - p1 * (1 - p1))
    d12 = sp.simplify(sp.diff(p1, z2) + p1 * sp.exp(z2) / Z)
    assert d11 == 0 and d12 == 0
    p = sm.softmax(sm.SCORES)
    J = sm.softmax_jacobian(p)
    eps = 1e-6
    for j in range(3):
        d = np.zeros(3); d[j] = eps
        num = (sm.softmax(sm.SCORES + d) - sm.softmax(sm.SCORES - d)) / (2 * eps)
        assert np.allclose(J[:, j], num, atol=1e-6)


def test_cross_entropy_entropy_kl_identity():
    p, q = np.array([0.7, 0.2, 0.1]), sm.softmax(sm.SCORES)
    assert sm.cross_entropy(p, q) == pytest.approx(sm.entropy(p) + sm.kl(p, q))
    assert sm.kl(p, q) >= 0 and sm.kl(q, q) == pytest.approx(0.0)
    assert sm.cross_entropy(sm.one_hot(0, 3), q) == pytest.approx(-np.log(q[0]))
    assert sm.entropy([1 / 3] * 3) == pytest.approx(np.log(3))


def test_gradient_of_loss_is_p_minus_y():
    L, g = sm.loss_and_grad(sm.SCORES, 0)
    assert np.allclose(g, sm.numerical_grad(sm.SCORES, 0), atol=1e-6)
    assert np.allclose(g, sm.softmax(sm.SCORES) - sm.one_hot(0, 3))
    z1, z2, z3 = sp.symbols("z1 z2 z3")
    Lsym = -sp.log(sp.exp(z1) / (sp.exp(z1) + sp.exp(z2) + sp.exp(z3)))
    subs = dict(zip((z1, z2, z3), sm.SCORES))
    assert float(sp.diff(Lsym, z2).subs(subs)) == pytest.approx(g[1])


def test_descent_on_scores_raises_target_probability():
    rows = sm.train_scores(sm.SCORES, 0)
    losses = [r[2] for r in rows]
    assert all(a > b for a, b in zip(losses, losses[1:]))
    assert rows[-1][1][0] > rows[0][1][0]


def test_next_token_temperature():
    d1, d3 = sm.next_token_distribution(1.0), sm.next_token_distribution(0.3)
    assert d3["the"] > d1["the"] and d1["zebra"] < 0.01
    assert sum(d1.values()) == pytest.approx(1.0)


def test_classifier_learns_with_cross_entropy():
    X, labels, W, b, losses, accs = sm.make_classifier_demo()
    assert accs[0] < 0.5 and accs[-1] > 0.9
    assert losses[-1] < losses[0] / 2


def test_calibration_metric():
    assert sm.expected_calibration([0.9] * 10, [1] * 9 + [0]) == pytest.approx(0.0, abs=1e-9)
    assert sm.expected_calibration([0.9] * 10, [1] * 5 + [0] * 5) == pytest.approx(0.4)


def test_lesson_facts():
    f = sm.lesson_facts()
    assert f["p_pct"] == [66, 24, 10] and f["grad"][0] < 0 < f["grad"][1]
    assert f["clf_acc_end"] > 90 and f["nt_the_03"] > f["nt_the_1"]


# ---------------------------------------------------------------------------
# Inside real models (lesson v2): synthetic lung-nodule classifier and perplexity
# ---------------------------------------------------------------------------
from avirmath.mathcore import softmax as _sm


def test_nodule_demo_learns_and_outputs_probabilities():
    X, labels, diam, spic, W, b, losses, accs = _sm.make_nodule_demo()
    P = np.apply_along_axis(_sm.softmax, 1, X @ W.T + b)
    assert np.allclose(P.sum(1), 1.0)
    assert losses[-1] < losses[0] and accs[-1] > 0.85
    assert _sm.nodule_risk(18.0, 0.8, W, b) > 0.8
    assert _sm.nodule_risk(4.0, 0.2, W, b) < 0.2


def test_two_class_risk_is_sigmoid_of_score_difference():
    _, _, _, _, W, b, _, _ = _sm.make_nodule_demo()
    x = _sm._nodule_features([12.0], [0.6])[0]
    z = W @ x + b
    assert np.isclose(_sm.nodule_risk(12.0, 0.6, W, b), _sm.sigmoid(z[1] - z[0]))


def test_perplexity_of_uniform_is_vocab_size():
    V = _sm.GPT2_VOCAB
    assert np.isclose(_sm.perplexity(_sm.entropy(np.full(V, 1.0 / V))), V, rtol=1e-6)
    assert np.isclose(_sm.perplexity(0.0), 1.0)
    assert 20.0 < _sm.perplexity(_sm.PPL_EXAMPLE_CE) < 20.2



# ---------------------------------------------------------------------------
# Real dataset (lesson v3): UCI Dermatology, six diagnoses
# ---------------------------------------------------------------------------
def test_dermatology_loads_and_softmax_regression_learns_it():
    X, y = _sm.load_dermatology()
    assert X.shape == (366, 34) and set(y) == set(range(6)) and not np.isnan(X).any()
    D = _sm.dermatology_model()
    assert np.allclose(D["P"].sum(1), 1.0) and D["hist"][-1][0] < D["hist"][0][0]
    assert D["acc"] > 0.85 and D["top2"] >= D["acc"] and D["cm"].sum() == D["n_test"]
    assert D["cm"].trace() == round(D["acc"] * D["n_test"])


def test_the_failure_demonstrations_behave_as_the_lesson_says():
    D = _sm.dermatology_model()
    assert D["acc_noisy"] < D["acc"]                                  # 20% flipped labels hurt
    assert D["conf_over"] >= D["conf_mean"] and D["ece_over"] >= D["ece"]   # unregularised training is more over-confident
    assert D["acc_clinical"] < D["acc"]                               # without the biopsy findings accuracy falls
    assert D["T"] > 0 and 0 <= D["ece_T"] <= 1


def test_multiclass_ece_and_confusion_definitions():
    P = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]]); y = np.array([0, 0, 1])
    assert np.isclose(_sm.ece_multiclass(P, y, bins=2), (2 / 3) * abs(0.85 - 1.0) + (1 / 3) * abs(0.7 - 1.0))
    cm = _sm.confusion_matrix([0, 1, 1, 2], [0, 1, 2, 2], 3)
    assert cm.tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]
