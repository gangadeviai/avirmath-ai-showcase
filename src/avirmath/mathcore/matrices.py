"""Matrices as transformations — the single source of mathematical truth for the lesson.

Sections
    1. A matrix moves space: M v is a combination of the columns
    2. Composition is multiplication, and order matters
    3. Determinant = area factor; inverse undoes
    4. Eigenvectors: directions that only stretch; the bowl's curvatures (lesson 1 callback)
    5. Where AI uses it: a layer is W x + b; attention is Q K^T
    6. Real time and industry: rotation matrices on the GPU, PageRank (1998), SVD compression
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray

M0 = np.array([[2.0, 1.0], [0.0, 1.0]])        # the worked matrix: stretch x by 2 and shear
V0 = np.array([1.0, 2.0])
R90 = np.array([[0.0, -1.0], [1.0, 0.0]])       # rotation by 90 degrees
S2 = np.array([[2.0, 0.0], [0.0, 1.0]])         # stretch x by 2
A_SYM = np.array([[2.0, 1.0], [1.0, 2.0]])      # symmetric: eigenvalues 3 and 1
H_BOWL = np.array([[2.0, 0.0], [0.0, 6.0]])     # Hessian of f = x^2 + 3y^2 (lesson 1)


# ---------------------------------------------------------------------------
# 1. M v
# ---------------------------------------------------------------------------
def apply(M, v) -> Array:
    return np.asarray(M, dtype=float) @ np.asarray(v, dtype=float)


def as_column_combination(M, v) -> Array:
    """v1 * (first column) + v2 * (second column): the same thing, seen as a recipe."""
    M, v = np.asarray(M, dtype=float), np.asarray(v, dtype=float)
    return sum(v[j] * M[:, j] for j in range(M.shape[1]))


def transformed_grid(M, n: int = 5, step: float = 1.0):
    """Line segments of the integer grid after M, for drawing. Returns list of (p, q) endpoint pairs."""
    segs = []
    r = n * step
    for k in np.arange(-n, n + 1) * step:
        segs.append((apply(M, [k, -r]), apply(M, [k, r])))
        segs.append((apply(M, [-r, k]), apply(M, [r, k])))
    return segs


# ---------------------------------------------------------------------------
# 2. Composition
# ---------------------------------------------------------------------------
def compose(first, then) -> Array:
    """Apply `first`, then `then`:  then @ first."""
    return np.asarray(then, dtype=float) @ np.asarray(first, dtype=float)


def commutator_gap(A, B) -> float:
    return float(np.abs(A @ B - B @ A).max())


# ---------------------------------------------------------------------------
# 3. Determinant and inverse
# ---------------------------------------------------------------------------
def det2(M) -> float:
    M = np.asarray(M, dtype=float)
    return float(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])


def polygon_area(points) -> float:
    """Shoelace formula; signed."""
    P = np.asarray(points, dtype=float)
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def unit_square_image(M) -> Array:
    return np.array([apply(M, p) for p in ([0, 0], [1, 0], [1, 1], [0, 1])])


def inverse2(M) -> Array:
    M = np.asarray(M, dtype=float)
    d = det2(M)
    if abs(d) < 1e-12:
        raise ZeroDivisionError("singular: the matrix flattens space, nothing can undo it")
    return np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]) / d


# ---------------------------------------------------------------------------
# 4. Eigenvectors
# ---------------------------------------------------------------------------
def eigen_sym(A):
    """Eigenvalues (descending) and unit eigenvectors (columns) of a symmetric matrix."""
    w, V = np.linalg.eigh(np.asarray(A, dtype=float))
    order = np.argsort(-w)
    return w[order], V[:, order]


def is_eigenpair(A, v, lam, tol=1e-9) -> bool:
    return bool(np.allclose(apply(A, v), lam * np.asarray(v, dtype=float), atol=tol))


def quadratic_form(A, x, y):
    """q(x, y) = [x y] A [x y]^T  — the bowl whose curvatures are the eigenvalues."""
    A = np.asarray(A, dtype=float)
    return A[0, 0] * x * x + 2 * A[0, 1] * x * y + A[1, 1] * y * y


def lr_limit(H) -> float:
    """Gradient descent on a quadratic with Hessian H is stable iff eta < 2 / lambda_max."""
    return 2.0 / eigen_sym(H)[0][0]


# ---------------------------------------------------------------------------
# 5. Where AI uses it
# ---------------------------------------------------------------------------
def layer(W, x, b):
    return apply(W, x) + np.asarray(b, dtype=float)


def attention_matrix(Q, K) -> Array:
    """All query-key dot products at once: Q K^T / sqrt(d)."""
    Q, K = np.asarray(Q, dtype=float), np.asarray(K, dtype=float)
    return Q @ K.T / np.sqrt(Q.shape[1])


# ---------------------------------------------------------------------------
# 6. Real time and industry
# ---------------------------------------------------------------------------
def rotation_z(theta: float) -> Array:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_x(theta: float) -> Array:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


CUBE = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
CUBE_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(CUBE[i] != CUBE[j]) == 1]

# A six-page web. LINKS[i] = pages that page i links to.
LINKS = {0: [1, 2], 1: [2], 2: [0], 3: [2, 4], 4: [2, 5], 5: [2, 4]}
DAMPING = 0.85


def link_matrix(links=LINKS, n: int = 6) -> Array:
    """Column-stochastic matrix: M[j, i] = 1/outdegree(i) if i links to j."""
    M = np.zeros((n, n))
    for i, outs in links.items():
        for j in outs:
            M[j, i] = 1.0 / len(outs)
    return M


def google_matrix(links=LINKS, n: int = 6, d: float = DAMPING) -> Array:
    return d * link_matrix(links, n) + (1 - d) / n * np.ones((n, n))


def pagerank(links=LINKS, n: int = 6, d: float = DAMPING, iters: int = 50):
    """Power iteration: repeatedly apply the Google matrix. Returns (ranks, history)."""
    G = google_matrix(links, n, d)
    p = np.ones(n) / n
    hist = [p.copy()]
    for _ in range(iters):
        p = G @ p
        hist.append(p.copy())
    return p, np.array(hist)


def make_image(n: int = 48, seed: int = 0) -> Array:
    """SYNTHETIC grey image with structure (stripes, a disc, a gradient) so low rank works."""
    y, x = np.mgrid[0:n, 0:n] / n
    img = 0.5 + 0.3 * np.sin(8 * x) + 0.2 * (x + y) / 2
    img[(x - 0.6) ** 2 + (y - 0.4) ** 2 < 0.06] = 0.1
    img[(y > 0.75) & (x < 0.4)] = 0.9
    return np.clip(img, 0, 1)


def rank_k(img, k: int) -> Array:
    U, S, Vt = np.linalg.svd(np.asarray(img, dtype=float), full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vt[:k]


def rank_k_error(img, k: int) -> float:
    return float(np.linalg.norm(np.asarray(img) - rank_k(img, k)))


def storage(n: int, k: int) -> tuple[int, int]:
    """(numbers in the full n x n image, numbers in a rank-k factorisation)."""
    return n * n, k * (2 * n + 1)


# ---------------------------------------------------------------------------
# Facts the lesson may quote
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Inside real models (lesson v2): GPT-2 layer sizes, CT images, LoRA fine-tuning
# ---------------------------------------------------------------------------
GPT2 = {"layers": 12, "d": 768, "heads": 12, "mlp": 3072, "params_m": 124}   # Radford et al., 2019 (GPT-2 small)
CT_PX = 512                                                                    # standard CT slice: 512 x 512 pixels
LORA_RANK = 8                                                                  # Hu et al., 2021 (LoRA)


def lora_numbers(d: int, r: int) -> tuple[int, int]:
    """(numbers in a full d x d update, numbers in the rank-r factorisation B A with B: d x r, A: r x d)."""
    return d * d, 2 * d * r


def _model_facts() -> dict:
    full, low = lora_numbers(GPT2["d"], LORA_RANK)
    return {
        "gpt2_layers": GPT2["layers"], "gpt2_d": GPT2["d"], "gpt2_heads": GPT2["heads"], "gpt2_mlp": GPT2["mlp"],
        "gpt2_params_m": GPT2["params_m"], "gpt2_layer_numbers": GPT2["d"] * GPT2["mlp"], "gpt2_head_dim": GPT2["d"] // GPT2["heads"],
        "ct_px": CT_PX, "ct_numbers": CT_PX * CT_PX,
        "lora_full": full, "lora_r": LORA_RANK, "lora_numbers": low, "lora_ratio": full // low,
    }


def lesson_facts() -> dict:
    w, V = eigen_sym(A_SYM)
    hw, _ = eigen_sym(H_BOWL)
    ranks, hist = pagerank()
    img = make_image()
    full, low = storage(48, 6)
    seq = compose(S2, R90); seq2 = compose(R90, S2)
    return {
        "m00": M0[0, 0], "m01": M0[0, 1], "m10": M0[1, 0], "m11": M0[1, 1],
        "vx": V0[0], "vy": V0[1], "mvx": apply(M0, V0)[0], "mvy": apply(M0, V0)[1],
        "det": det2(M0), "area_before": 1.0, "area_after": abs(polygon_area(unit_square_image(M0))),
        "lam1": w[0], "lam2": w[1], "ev1": [round(x, 3) for x in V[:, 0]], "ev2": [round(x, 3) for x in V[:, 1]],
        "bowl_lam_max": hw[0], "bowl_lam_min": hw[1], "bowl_lr_limit": round(lr_limit(H_BOWL), 3),
        "order_gap": commutator_gap(S2, R90), "seq_rs": seq.round(2).tolist(), "seq_sr": seq2.round(2).tolist(),
        "pr": [round(float(r), 3) for r in ranks], "pr_top": int(np.argmax(ranks)), "pr_iters_to_1e-6": int(np.argmax(np.linalg.norm(hist - ranks, axis=1) < 1e-6)),
        "damping": DAMPING, "img_n": 48, "rank_k": 6, "storage_full": full, "storage_low": low,
        "rank_err_1": round(rank_k_error(img, 1), 2), "rank_err_6": round(rank_k_error(img, 6), 2), "rank_err_20": round(rank_k_error(img, 20), 3),
        # Netflix Prize (2006-2009), from the public record: ratings matrix size and the target
        "nf_users": 480_189, "nf_movies": 17_770, "nf_ratings_m": 100.48, "nf_cells_b": round(480_189 * 17_770 / 1e9, 1),
        "nf_filled_pct": round(100.48e6 / (480_189 * 17_770) * 100, 1), "nf_rmse_from": 0.9514, "nf_rmse_to": 0.8572,
        **_model_facts(),
    }
