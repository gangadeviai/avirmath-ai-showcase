"""Vectors and the dot product — the single source of mathematical truth for the lesson.

Sections
    1. Vectors: arithmetic, length, unit vectors
    2. The dot product: algebraic and geometric definitions, and why they agree
    3. Projection (the shadow) and perpendicularity
    4. Cosine similarity and Salton's document vectors (1975)
    5. Where AI uses it: embeddings and analogies, attention scores, the neuron
    6. Real time and industry: Lambert shading (1760), semantic retrieval, mechanical work
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray

A0 = np.array([3.0, 4.0])
B0 = np.array([4.0, -3.0])
C0 = np.array([2.0, 1.0])
U3 = np.array([1.0, 2.0, 2.0])      # length 3 in 3-D


# ---------------------------------------------------------------------------
# 1. Vectors
# ---------------------------------------------------------------------------
def norm(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(np.sqrt(np.sum(v * v)))


def unit(v) -> Array:
    v = np.asarray(v, dtype=float)
    return v / norm(v)


# ---------------------------------------------------------------------------
# 2. The dot product
# ---------------------------------------------------------------------------
def dot_algebraic(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sum(a * b))


def angle_between(a, b) -> float:
    c = dot_algebraic(a, b) / (norm(a) * norm(b))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def dot_geometric(a, b, theta: float | None = None) -> float:
    """|a| |b| cos(theta). theta defaults to the true angle, so this must equal dot_algebraic."""
    th = angle_between(a, b) if theta is None else theta
    return norm(a) * norm(b) * float(np.cos(th))


def law_of_cosines_residual(a, b) -> float:
    """|a-b|^2 = |a|^2 + |b|^2 - 2 a.b  — the identity that makes the two definitions one."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return norm(a - b) ** 2 - (norm(a) ** 2 + norm(b) ** 2 - 2 * dot_algebraic(a, b))


# ---------------------------------------------------------------------------
# 3. Projection
# ---------------------------------------------------------------------------
def scalar_projection(a, onto) -> float:
    """Signed length of a's shadow on `onto`: a . unit(onto)."""
    return dot_algebraic(a, unit(onto))


def vector_projection(a, onto) -> Array:
    u = unit(onto)
    return scalar_projection(a, onto) * u


def rotated(v, theta: float) -> Array:
    """Rotate a 2-D vector by theta (used for the shadow sweep)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]]) @ np.asarray(v, dtype=float)


# ---------------------------------------------------------------------------
# 4. Cosine similarity and document vectors
# ---------------------------------------------------------------------------
def cosine_similarity(a, b) -> float:
    return dot_algebraic(a, b) / (norm(a) * norm(b))


VOCAB = ("gradient", "descent", "loss", "chain", "rule", "weight")
DOC_A = "the gradient points uphill so gradient descent steps against the gradient to lower the loss"
DOC_B = "the chain rule gives the gradient of the loss with respect to every weight"
DOC_C = "the chain rule the chain rule the chain rule"


def bag_of_words(text: str, vocab=VOCAB) -> Array:
    words = text.lower().split()
    return np.array([words.count(v) for v in vocab], dtype=float)


# ---------------------------------------------------------------------------
# 5. Where AI uses it
# ---------------------------------------------------------------------------
# A tiny hand-built embedding space in which the famous analogy holds exactly.
# Coordinates: (royalty, gender, age). Real embeddings have hundreds of coordinates; the arithmetic is identical.
EMBED = {
    "king": np.array([1.0, 1.0, 0.0]), "queen": np.array([1.0, -1.0, 0.0]),
    "man": np.array([0.0, 1.0, 0.0]), "woman": np.array([0.0, -1.0, 0.0]),
    "prince": np.array([1.0, 1.0, -1.0]), "princess": np.array([1.0, -1.0, -1.0]),
    "boy": np.array([0.0, 1.0, -1.0]), "girl": np.array([0.0, -1.0, -1.0]),
}


def nearest_by_cosine(v, exclude=()) -> list[tuple[str, float]]:
    return sorted(((w, cosine_similarity(v, e)) for w, e in EMBED.items() if w not in exclude), key=lambda t: -t[1])


def analogy(a: str, b: str, c: str) -> list[tuple[str, float]]:
    """a - b + c, ranked by cosine. king - man + woman -> queen."""
    return nearest_by_cosine(EMBED[a] - EMBED[b] + EMBED[c], exclude=(a, b, c))


def attention_scores(q, keys, d_k: int | None = None) -> Array:
    """Scaled dot-product attention scores (before softmax): q . k_j / sqrt(d_k)."""
    q, K = np.asarray(q, dtype=float), np.asarray(keys, dtype=float)
    d = q.size if d_k is None else d_k
    return K @ q / np.sqrt(d)


def neuron(w, x, b: float) -> float:
    return dot_algebraic(w, x) + b


# ---------------------------------------------------------------------------
# 6. Real time and industry
# ---------------------------------------------------------------------------
def lambert(normal, light) -> float:
    """Lambert's cosine law (Photometria, 1760): brightness = max(0, n . l) for unit vectors."""
    return max(0.0, dot_algebraic(unit(normal), unit(light)))


def sphere_shading(light, n: int = 41) -> Array:
    """Brightness of a unit sphere seen from +z, lit from `light`. NaN outside the disc."""
    xs = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xs, xs)
    Z2 = 1 - X ** 2 - Y ** 2
    img = np.full_like(X, np.nan)
    inside = Z2 >= 0
    N = np.stack([X[inside], Y[inside], np.sqrt(Z2[inside])], axis=1)
    img[inside] = np.clip(N @ unit(light), 0, None)
    return img


def retrieve(query, items: Array, k: int = 3) -> list[tuple[int, float]]:
    """Semantic retrieval: rank items by cosine similarity to the query (what a vector database does)."""
    q = unit(query)
    sims = (items / np.linalg.norm(items, axis=1, keepdims=True)) @ q
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


def make_catalogue(n: int = 10_000, d: int = 64, seed: int = 0) -> Array:
    """SYNTHETIC item embeddings in a few clusters, to time a real retrieval."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(8, d))
    labels = rng.integers(0, 8, n)
    return centres[labels] + 0.6 * rng.normal(size=(n, d))


def work_done(force, displacement) -> float:
    """Mechanical work W = F . d (joules when newtons and metres)."""
    return dot_algebraic(force, displacement)


# ---------------------------------------------------------------------------
# Facts the lesson may quote
# ---------------------------------------------------------------------------
def lesson_facts() -> dict:
    a, b, c = A0, B0, C0
    docA, docB, docC = bag_of_words(DOC_A), bag_of_words(DOC_B), bag_of_words(DOC_C)
    top = analogy("king", "man", "woman")
    light = np.array([1.0, 1.0, 1.0])
    q = np.array([1.0, 0.0, 1.0, 0.0])
    keys = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    scores = attention_scores(q, keys)
    return {
        "ax": a[0], "ay": a[1], "bx": b[0], "by": b[1], "cx": c[0], "cy": c[1],
        "len_a": norm(a), "len_b": norm(b), "len_u3": norm(U3),
        "a_dot_b": dot_algebraic(a, b), "a_dot_c": dot_algebraic(a, c),
        "angle_ac_deg": round(float(np.degrees(angle_between(a, c))), 1),
        "cos_ac": round(float(np.cos(angle_between(a, c))), 3),
        "proj_a_on_c": round(scalar_projection(a, c), 3),
        "cos_AB": round(cosine_similarity(docA, docB), 2), "cos_AC": round(cosine_similarity(docA, docC), 2),
        "cos_BC": round(cosine_similarity(docB, docC), 2),
        "analogy_top": top[0][0], "analogy_cos": round(top[0][1], 2),
        "att_scores": [round(float(s), 2) for s in scores],
        "lambert_face": round(lambert([0, 0, 1], light), 3), "lambert_side": round(lambert([1, 0, 0], light), 3),
        "lambert_back": lambert([0, 0, -1], light),
        "catalogue_n": 10_000, "catalogue_d": 64,
        "work_j": work_done([10.0, 0.0], [3.0, 4.0]),
    }
