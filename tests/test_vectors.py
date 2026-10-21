"""Independent validation of the vectors & dot product mathematics."""

import numpy as np
import pytest
import sympy as sp

from avirmath.mathcore import vectors as vc


def test_lengths_by_pythagoras():
    assert vc.norm([3, 4]) == 5
    assert vc.norm(vc.U3) == 3
    assert vc.norm(vc.unit([7, -2, 5])) == pytest.approx(1.0)


@pytest.mark.parametrize("a,b", [([3, 4], [4, -3]), ([3, 4], [2, 1]), ([1, 2, 2], [2, -1, 3]), ([0.3, -1.2, 5, 2], [1, 1, 1, 1])])
def test_algebraic_and_geometric_definitions_agree(a, b):
    assert vc.dot_algebraic(a, b) == pytest.approx(vc.dot_geometric(a, b), abs=1e-9)
    assert vc.law_of_cosines_residual(a, b) == pytest.approx(0.0, abs=1e-9)


def test_law_of_cosines_symbolically():
    a1, a2, b1, b2 = sp.symbols("a1 a2 b1 b2", real=True)
    lhs = (a1 - b1) ** 2 + (a2 - b2) ** 2
    rhs = (a1 ** 2 + a2 ** 2) + (b1 ** 2 + b2 ** 2) - 2 * (a1 * b1 + a2 * b2)
    assert sp.simplify(lhs - rhs) == 0


def test_worked_examples():
    assert vc.dot_algebraic(vc.A0, vc.B0) == 0                       # perpendicular
    assert vc.angle_between(vc.A0, vc.B0) == pytest.approx(np.pi / 2)
    assert vc.dot_algebraic(vc.A0, vc.C0) == 10
    assert vc.scalar_projection(vc.A0, vc.C0) == pytest.approx(10 / np.sqrt(5))


def test_projection_sign_and_shadow():
    assert vc.scalar_projection([1, 1], [1, 0]) == pytest.approx(1.0)
    assert vc.scalar_projection([-1, 1], [1, 0]) == pytest.approx(-1.0)
    assert vc.scalar_projection([0, 1], [1, 0]) == pytest.approx(0.0)
    p = vc.vector_projection(vc.A0, vc.C0)
    assert vc.dot_algebraic(vc.A0 - p, vc.C0) == pytest.approx(0.0, abs=1e-9)   # the residual is perpendicular


def test_rotation_sweep_matches_cosine():
    base = np.array([2.0, 0.0])
    for th in np.linspace(0, np.pi, 13):
        assert vc.dot_algebraic(vc.rotated(base, th), [1, 0]) == pytest.approx(2 * np.cos(th), abs=1e-9)


def test_cosine_similarity_ignores_length():
    assert vc.cosine_similarity([1, 2], [3, 6]) == pytest.approx(1.0)
    assert vc.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert vc.cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_document_vectors_rank_as_claimed():
    A, B, C = (vc.bag_of_words(d) for d in (vc.DOC_A, vc.DOC_B, vc.DOC_C))
    assert vc.cosine_similarity(A, B) > vc.cosine_similarity(A, C)   # A is about gradients; C is only "chain rule"
    assert vc.cosine_similarity(B, C) > vc.cosine_similarity(A, C)


def test_king_minus_man_plus_woman_is_queen():
    top = vc.analogy("king", "man", "woman")
    assert top[0][0] == "queen" and top[0][1] == pytest.approx(1.0)


def test_attention_scores_are_scaled_dot_products():
    q = np.array([1.0, 0.0, 1.0, 0.0]); K = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    assert np.allclose(vc.attention_scores(q, K), [2 / 2, 0.0])          # d_k = 4 -> divide by 2


def test_neuron_is_a_dot_product():
    assert vc.neuron([0.5, -1.0, 2.0], [2.0, 1.0, 0.5], 0.3) == pytest.approx(1.0 - 1.0 + 1.0 + 0.3)


def test_lambert_law():
    assert vc.lambert([0, 0, 1], [0, 0, 1]) == pytest.approx(1.0)
    assert vc.lambert([0, 0, 1], [1, 0, 1]) == pytest.approx(np.cos(np.pi / 4))
    assert vc.lambert([0, 0, 1], [0, 0, -1]) == 0.0                    # facing away: clamped, not negative
    img = vc.sphere_shading([0, 0, 1])
    assert np.nanmax(img) == pytest.approx(1.0) and np.nanmin(img) >= 0


def test_retrieval_returns_the_nearest_by_cosine():
    items = vc.make_catalogue(n=2000, d=16)
    q = items[123] + 0.05 * np.random.default_rng(1).normal(size=16)
    top = vc.retrieve(q, items, k=3)
    assert top[0][0] == 123 and top[0][1] > 0.95


def test_work_done():
    assert vc.work_done([10, 0], [3, 4]) == 30                          # only the component along the force counts


def test_lesson_facts():
    f = vc.lesson_facts()
    assert f["a_dot_b"] == 0 and f["len_a"] == 5 and f["analogy_top"] == "queen"
    assert f["lambert_back"] == 0 and f["work_j"] == 30
