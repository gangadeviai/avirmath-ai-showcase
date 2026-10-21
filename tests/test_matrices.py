"""Independent validation of the matrices-as-transformations mathematics."""

import numpy as np
import pytest
import sympy as sp

from avirmath.mathcore import matrices as mx


def test_matrix_vector_is_a_combination_of_columns():
    assert np.allclose(mx.apply(mx.M0, mx.V0), [4, 2])
    assert np.allclose(mx.apply(mx.M0, mx.V0), mx.as_column_combination(mx.M0, mx.V0))
    assert np.allclose(mx.apply(mx.M0, [1, 0]), mx.M0[:, 0]) and np.allclose(mx.apply(mx.M0, [0, 1]), mx.M0[:, 1])


def test_linearity():
    rng = np.random.default_rng(0)
    u, v, M = rng.normal(size=2), rng.normal(size=2), rng.normal(size=(2, 2))
    assert np.allclose(mx.apply(M, 3 * u + 2 * v), 3 * mx.apply(M, u) + 2 * mx.apply(M, v))


def test_composition_is_matrix_product_and_order_matters():
    v = np.array([1.0, 2.0])
    assert np.allclose(mx.apply(mx.compose(mx.S2, mx.R90), v), mx.apply(mx.R90, mx.apply(mx.S2, v)))
    assert not np.allclose(mx.compose(mx.S2, mx.R90), mx.compose(mx.R90, mx.S2))
    assert mx.commutator_gap(mx.S2, mx.R90) > 0.5


def test_determinant_is_the_area_factor():
    for M in (mx.M0, mx.R90, mx.S2, np.array([[1, 2], [2, 4]])):
        assert abs(mx.polygon_area(mx.unit_square_image(M))) == pytest.approx(abs(mx.det2(M)))
    assert mx.det2([[1, 2], [2, 4]]) == 0


def test_determinant_symbolically():
    a, b, c, d = sp.symbols("a b c d")
    assert sp.simplify(sp.Matrix([[a, b], [c, d]]).det() - (a * d - b * c)) == 0


def test_inverse_undoes_and_singular_fails():
    assert np.allclose(mx.inverse2(mx.M0) @ mx.M0, np.eye(2))
    with pytest.raises(ZeroDivisionError):
        mx.inverse2([[1, 2], [2, 4]])


def test_eigenpairs_of_the_worked_symmetric_matrix():
    w, V = mx.eigen_sym(mx.A_SYM)
    assert np.allclose(w, [3, 1])
    assert mx.is_eigenpair(mx.A_SYM, V[:, 0], 3) and mx.is_eigenpair(mx.A_SYM, V[:, 1], 1)
    assert np.allclose(np.abs(V[:, 0]), [1, 1] / np.sqrt(2)) and np.allclose(np.abs(V[:, 1]), [1, 1] / np.sqrt(2))
    assert not mx.is_eigenpair(mx.A_SYM, [1, 0], 2)


def test_eigenvalues_symbolically():
    lam = sp.symbols("lam")
    A = sp.Matrix([[2, 1], [1, 2]])
    roots = sp.solve((A - lam * sp.eye(2)).det(), lam)
    assert sorted(float(r) for r in roots) == [1.0, 3.0]


def test_bowl_curvatures_are_hessian_eigenvalues_and_set_the_lr_limit():
    w, _ = mx.eigen_sym(mx.H_BOWL)
    assert np.allclose(w, [6, 2])
    assert mx.lr_limit(mx.H_BOWL) == pytest.approx(1 / 3)            # lesson 1's exact limit
    x, y = sp.symbols("x y")
    f = x ** 2 + 3 * y ** 2
    assert np.allclose(np.array(sp.hessian(f, (x, y)), dtype=float), mx.H_BOWL)
    assert mx.quadratic_form(mx.H_BOWL / 2, 1.5, -0.5) == pytest.approx(float(f.subs({x: 1.5, y: -0.5})))


def test_layer_and_attention_matrix():
    W, x, b = np.array([[1, 2], [3, 4], [5, 6]]), np.array([1, -1]), np.array([0.5, 0.5, 0.5])
    assert np.allclose(mx.layer(W, x, b), [-0.5, -0.5, -0.5])
    Q = np.array([[1.0, 0, 1, 0], [0, 1, 0, 1]]); K = np.array([[1.0, 0, 1, 0], [1, 1, 0, 0]])
    A = mx.attention_matrix(Q, K)
    assert np.allclose(A, [[1.0, 0.5], [0.0, 0.5]])                   # q_i . k_j / sqrt(4)


def test_rotation_preserves_lengths_and_angles():
    R = mx.rotation_z(0.7) @ mx.rotation_x(-1.1)
    P = mx.CUBE @ R.T
    assert np.allclose(np.linalg.norm(P, axis=1), np.linalg.norm(mx.CUBE, axis=1))
    assert np.allclose(R.T @ R, np.eye(3)) and mx.det2(R[:2, :2]) <= 1.0
    assert len(mx.CUBE_EDGES) == 12


def test_pagerank_is_the_fixed_point_of_the_google_matrix():
    G = mx.google_matrix()
    assert np.allclose(G.sum(axis=0), 1.0)                            # column-stochastic
    ranks, hist = mx.pagerank()
    assert np.allclose(G @ ranks, ranks, atol=1e-9) and ranks.sum() == pytest.approx(1.0)
    w, V = np.linalg.eig(G)
    v = np.real(V[:, np.argmax(np.real(w))]); v = v / v.sum()
    assert np.allclose(ranks, v, atol=1e-6)                           # the principal eigenvector
    assert np.argmax(ranks) == 2                                      # page 2: everyone links to it


def test_low_rank_error_decreases_and_matches_discarded_singular_values():
    img = mx.make_image()
    S = np.linalg.svd(img, compute_uv=False)
    for k in (1, 6, 20):
        assert mx.rank_k_error(img, k) == pytest.approx(np.sqrt(np.sum(S[k:] ** 2)), rel=1e-6)
    assert mx.rank_k_error(img, 1) > mx.rank_k_error(img, 6) > mx.rank_k_error(img, 20)
    full, low = mx.storage(48, 6)
    assert low < full / 3


def test_lesson_facts():
    f = mx.lesson_facts()
    assert (f["mvx"], f["mvy"]) == (4, 2) and f["det"] == 2 and f["area_after"] == 2
    assert f["lam1"] == 3 and f["lam2"] == 1 and f["pr_top"] == 2 and f["bowl_lr_limit"] == pytest.approx(0.333)



# ---------------------------------------------------------------------------
# Inside real models (lesson v2)
# ---------------------------------------------------------------------------
from avirmath.mathcore import matrices as _mx


def test_gpt2_sizes_are_consistent():
    f = _mx.lesson_facts()
    assert f["gpt2_mlp"] == 4 * f["gpt2_d"] and f["gpt2_head_dim"] == 64
    assert f["gpt2_layer_numbers"] == f["gpt2_d"] * f["gpt2_mlp"] == 2_359_296
    assert f["ct_numbers"] == 512 * 512


def test_lora_storage_saving():
    full, low = _mx.lora_numbers(768, 8)
    assert full == 589_824 and low == 12_288 and full // low == 48
    d, r = 768, 8
    B, A = np.ones((d, r)), np.ones((r, d))
    assert (B @ A).shape == (d, d) and np.linalg.matrix_rank(B @ A) <= r
