"""Backpropagation — the single source of mathematical truth for the lesson.

Sections
    1. Chain rule on a composition  y = f(g(x))
    2. The worked network           x -> h = sigma(w1 x) -> yhat = w2 h -> L = (yhat - y)^2
    3. Cost: nudging vs backward sweep
    4. Matrix form for any depth     delta_l = (W_{l+1}^T delta_{l+1}) * sigma'(z_l),  dL/dW_l = delta_l h_{l-1}^T
    5. Vanishing gradients through sigmoid layers
    6. Industry example: a tiny digit classifier trained by backprop (synthetic 8x8 digits)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------------
# 1. Chain rule
# ---------------------------------------------------------------------------
def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def chain_rule_demo(x: float):
    """y = sin(x^2): dy/dx = cos(x^2) * 2x.  Returns (g', f', product)."""
    g_prime = 2.0 * x
    f_prime = float(np.cos(x * x))
    return g_prime, f_prime, g_prime * f_prime


# ---------------------------------------------------------------------------
# 2. The worked network (numbers the narrator quotes)
# ---------------------------------------------------------------------------
X0, Y0 = 1.5, 1.0          # one training example
W1_0, W2_0 = 0.8, -0.4     # starting weights
LR = 0.5


@dataclass(frozen=True)
class Pass:
    x: float; y: float; w1: float; w2: float
    z: float; h: float; yhat: float; loss: float
    dL_dyhat: float; dL_dw2: float; dL_dh: float; dh_dz: float; dL_dz: float; dL_dw1: float


def forward_backward(x=X0, y=Y0, w1=W1_0, w2=W2_0) -> Pass:
    z = w1 * x
    h = float(sigmoid(z))
    yhat = w2 * h
    loss = (yhat - y) ** 2
    dL_dyhat = 2.0 * (yhat - y)
    dL_dw2 = dL_dyhat * h                    # yhat = w2 h
    dL_dh = dL_dyhat * w2                    # shared by everything upstream of h
    dh_dz = float(sigmoid_prime(z))
    dL_dz = dL_dh * dh_dz
    dL_dw1 = dL_dz * x                       # z = w1 x
    return Pass(x, y, w1, w2, z, h, yhat, loss, dL_dyhat, dL_dw2, dL_dh, dh_dz, dL_dz, dL_dw1)


def loss_surface(w1, w2, x=X0, y=Y0):
    """L(w1, w2) for the worked network — the 3D surface in the video."""
    return (np.asarray(w2) * sigmoid(np.asarray(w1) * x) - y) ** 2


def train_worked(steps: int = 12, lr: float = LR, w1=W1_0, w2=W2_0) -> Array:
    """Gradient descent on the worked network using backprop gradients. Returns (steps+1, 3): w1, w2, loss."""
    rows = []
    for _ in range(steps + 1):
        p = forward_backward(w1=w1, w2=w2)
        rows.append((w1, w2, p.loss))
        w1, w2 = w1 - lr * p.dL_dw1, w2 - lr * p.dL_dw2
    return np.array(rows)


# ---------------------------------------------------------------------------
# 3. Cost of nudging vs one backward sweep
# ---------------------------------------------------------------------------
def nudge_cost(n_weights: int) -> int:
    """Finite differences: one forward pass per weight (plus the base pass)."""
    return n_weights + 1


def backprop_cost(n_weights: int) -> int:
    """One forward pass + one backward pass, regardless of the number of weights (in passes)."""
    return 2


# ---------------------------------------------------------------------------
# 4. General matrix form (any depth), with an independent numerical check
# ---------------------------------------------------------------------------
def init_mlp(sizes, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, scale / np.sqrt(a), (b, a)) for a, b in zip(sizes[:-1], sizes[1:])]


def mlp_forward(Ws, x):
    """Sigmoid hidden layers, linear output. Returns (zs, hs) with hs[0] = x."""
    hs, zs = [np.asarray(x, dtype=float)], []
    for i, W in enumerate(Ws):
        z = W @ hs[-1]
        zs.append(z)
        hs.append(z if i == len(Ws) - 1 else sigmoid(z))
    return zs, hs


def mlp_loss(Ws, x, y):
    _, hs = mlp_forward(Ws, x)
    return float(np.sum((hs[-1] - y) ** 2))


def mlp_backprop(Ws, x, y):
    """Returns (grads, deltas). delta_l = dL/dz_l, exactly the three lines shown in the video."""
    zs, hs = mlp_forward(Ws, x)
    L = len(Ws)
    deltas = [None] * L
    deltas[-1] = 2.0 * (hs[-1] - y)                                  # linear output: sigma' = 1
    for l in range(L - 2, -1, -1):
        deltas[l] = (Ws[l + 1].T @ deltas[l + 1]) * sigmoid_prime(zs[l])
    grads = [np.outer(deltas[l], hs[l]) for l in range(L)]
    return grads, deltas


def mlp_numerical_grads(Ws, x, y, eps=1e-6):
    grads = []
    for l, W in enumerate(Ws):
        g = np.zeros_like(W)
        for idx in np.ndindex(W.shape):
            Wp = [w.copy() for w in Ws]; Wm = [w.copy() for w in Ws]
            Wp[l][idx] += eps; Wm[l][idx] -= eps
            g[idx] = (mlp_loss(Wp, x, y) - mlp_loss(Wm, x, y)) / (2 * eps)
        grads.append(g)
    return grads


# ---------------------------------------------------------------------------
# 5. Vanishing gradients
# ---------------------------------------------------------------------------
SIGMOID_PRIME_MAX = 0.25


def delta_norms_by_layer(depth: int = 6, width: int = 16, seed: int = 1) -> Array:
    """||delta_l|| for a deep sigmoid network with unit-variance init: shrinks towards the input."""
    Ws = init_mlp([width] * (depth + 1), seed=seed)
    rng = np.random.default_rng(seed)
    x, y = rng.normal(size=width), rng.normal(size=width)
    _, deltas = mlp_backprop(Ws, x, y)
    return np.array([np.linalg.norm(d) for d in deltas])


# ---------------------------------------------------------------------------
# 6. Industry example: digits, trained by backprop
# ---------------------------------------------------------------------------
def make_digits(n_per_class: int = 60, seed: int = 3):
    """SYNTHETIC 8x8 digit-like images (0-9) from fixed stroke templates + noise. Returns (X, labels)."""
    rng = np.random.default_rng(seed)
    T = np.zeros((10, 8, 8))
    T[0, 1:7, [1, 6]] = 1; T[0, [1, 6], 1:7] = 1
    T[1, 1:7, 4] = 1
    T[2, 1, 1:7] = 1; T[2, 1:4, 6] = 1; T[2, 4, 1:7] = 1; T[2, 4:7, 1] = 1; T[2, 6, 1:7] = 1
    T[3, [1, 4, 6], 1:7] = 1; T[3, 1:7, 6] = 1
    T[4, 1:5, 1] = 1; T[4, 4, 1:7] = 1; T[4, 1:7, 6] = 1
    T[5, 1, 1:7] = 1; T[5, 1:4, 1] = 1; T[5, 4, 1:7] = 1; T[5, 4:7, 6] = 1; T[5, 6, 1:7] = 1
    T[6, 1:7, 1] = 1; T[6, [1, 4, 6], 1:7] = 1; T[6, 4:7, 6] = 1
    T[7, 1, 1:7] = 1; T[7, 1:7, 6] = 1
    T[8, [1, 4, 6], 1:7] = 1; T[8, 1:7, [1, 6]] = 1
    T[9, [1, 4, 6], 1:7] = 1; T[9, 1:4, 1] = 1; T[9, 1:7, 6] = 1
    X, labels = [], []
    for d in range(10):
        for _ in range(n_per_class):
            img = T[d].copy()
            img = img * rng.uniform(0.7, 1.0) + rng.normal(0, 0.25, (8, 8))   # heavy pixel noise; no shift (shifts make 3/8/9 identical at 8x8)
            X.append(img.ravel()); labels.append(d)
    return np.array(X), np.array(labels)


@dataclass
class DigitsLog:
    losses: Array
    accuracies: Array
    Ws: list


def train_digits(epochs: int = 40, lr: float = 0.05, hidden: int = 32, seed: int = 0):
    X, labels = make_digits()
    Y = np.eye(10)[labels]
    rng = np.random.default_rng(seed)
    n = len(X)
    test = rng.random(n) < 0.25
    Ws = init_mlp([64, hidden, 10], seed=seed)
    losses, accs = [], []
    for _ in range(epochs):
        for i in rng.permutation(np.where(~test)[0]):
            grads, _ = mlp_backprop(Ws, X[i], Y[i])
            Ws = [W - lr * g for W, g in zip(Ws, grads)]
        preds = np.array([np.argmax(mlp_forward(Ws, x)[1][-1]) for x in X[test]])
        accs.append(float(np.mean(preds == labels[test])))
        losses.append(float(np.mean([mlp_loss(Ws, X[i], Y[i]) for i in np.where(test)[0]])))
    return DigitsLog(np.array(losses), np.array(accs), Ws)


# ---------------------------------------------------------------------------
# Facts the lesson may quote
# ---------------------------------------------------------------------------
def lesson_facts() -> dict:
    p = forward_backward()
    log = train_digits()
    traj = train_worked()
    norms = delta_norms_by_layer()
    return {
        "x": X0, "y": Y0, "w1": W1_0, "w2": W2_0, "lr": LR,
        "z": round(p.z, 2), "h": round(p.h, 3), "yhat": round(p.yhat, 3), "loss": round(p.loss, 3),
        "dL_dyhat": round(p.dL_dyhat, 3), "dL_dw2": round(p.dL_dw2, 3), "dL_dh": round(p.dL_dh, 3),
        "dh_dz": round(p.dh_dz, 3), "dL_dw1": round(p.dL_dw1, 3),
        "w1_new": round(W1_0 - LR * p.dL_dw1, 3), "w2_new": round(W2_0 - LR * p.dL_dw2, 3),
        "loss_after": round(float(traj[1, 2]), 3), "loss_after_12": round(float(traj[-1, 2]), 4),
        "nudge_cost_1k": nudge_cost(1000), "nudge_cost_1b": nudge_cost(10 ** 9),
        "sigmoid_prime_max": SIGMOID_PRIME_MAX,
        "vanish_ratio": int(round(norms[-1] / norms[0])),      # how many times smaller at the input layer
        "digits_n": 600, "digits_epochs": 40,
        "digits_acc_start": round(float(log.accuracies[0]) * 100), "digits_acc_final": round(float(log.accuracies[-1]) * 100),
        "lenet_images": 9298, "lenet_year": 1989,
    }
