"""Gradient descent — the single source of mathematical truth for the lesson.

Everything the learner sees (story numbers, equations, the web experiment, the
animation, quiz answers) must come from the functions in this file. No graphics
code lives here, and nothing here is approximate "for show".

Sections
    1. The generic algorithm
    2. Chai stall, one knob      L(s)    = (s - s*)^2
    3. Chai stall, two knobs     L(s, t) = a(s-s*)^2 + b(t-t*)^2      (elongated bowl)
    4. Industry example          delivery-time (ETA) prediction, linear model, mini-batch SGD
    5. Learning-rate schedule    linear warmup + cosine decay (as used when training LLMs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------------
# 1. The generic algorithm
# ---------------------------------------------------------------------------
def gradient_descent(grad: Callable[[Array], Array], x0, lr: float, steps: int) -> Array:
    """Iterate x <- x - lr * grad(x). Returns the whole path, shape (steps + 1, dim)."""
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    path = [x.copy()]
    for _ in range(steps):
        x = x - lr * np.atleast_1d(grad(x))
        path.append(x.copy())
    return np.array(path)


def numerical_gradient(f: Callable[[Array], float], x, eps: float = 1e-6) -> Array:
    """Central finite differences. Used ONLY to verify analytic gradients."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    g = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        g[i] = (f(x + step) - f(x - step)) / (2 * eps)
    return g


# ---------------------------------------------------------------------------
# 2. Chai stall, one knob: grams of sugar per cup
# ---------------------------------------------------------------------------
SUGAR_BEST = 8.0    # grams per cup her customers like most (unknown to her)
SUGAR_START = 2.0   # where she starts


def chai_loss(s, s_best: float = SUGAR_BEST):
    """Customer unhappiness as a function of sugar: L(s) = (s - s*)^2."""
    return (np.asarray(s, dtype=float) - s_best) ** 2


def chai_grad(s, s_best: float = SUGAR_BEST):
    """dL/ds = 2 (s - s*). Negative when there is too little sugar -> add more."""
    return 2.0 * (np.asarray(s, dtype=float) - s_best)


def chai_path(lr: float, steps: int = 12, s0: float = SUGAR_START) -> Array:
    """Sugar level day by day, shape (steps + 1,)."""
    return gradient_descent(chai_grad, [s0], lr, steps)[:, 0]


def chai_error_ratio(lr: float) -> float:
    """Each day the distance to the best recipe is multiplied by (1 - 2*lr).

    Derivation: s' - s* = (s - s*) - lr * 2 (s - s*) = (1 - 2 lr)(s - s*).
    """
    return 1.0 - 2.0 * lr


def chai_regime(lr: float) -> str:
    """Behaviour of the chai recipe for a given learning rate (exact, from the ratio)."""
    r = chai_error_ratio(lr)
    if lr <= 0:
        return "stuck"
    if abs(r) > 1:
        return "diverges"
    if abs(r) == 1:
        return "oscillates forever"
    if r == 0:
        return "one step"
    return "converges smoothly" if r > 0 else "converges with overshoot"


CHAI_LR_MAX = 1.0  # |1 - 2 lr| < 1  <=>  0 < lr < 1


# ---------------------------------------------------------------------------
# 3. Chai stall, two knobs: sugar (s) and tea strength (t) -> an elongated bowl
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Bowl:
    """L(p) = a (s - s*)^2 + b (t - t*)^2 with p = (s, t)."""

    a: float = 1.0
    b: float = 8.0          # customers are 8x more sensitive to tea strength
    s_best: float = 8.0
    t_best: float = 3.0

    @property
    def best(self) -> Array:
        return np.array([self.s_best, self.t_best])

    def loss(self, p) -> float:
        s, t = np.asarray(p, dtype=float)
        return self.a * (s - self.s_best) ** 2 + self.b * (t - self.t_best) ** 2

    def grad(self, p) -> Array:
        s, t = np.asarray(p, dtype=float)
        return np.array([2 * self.a * (s - self.s_best), 2 * self.b * (t - self.t_best)])

    @property
    def curvatures(self) -> Array:
        """Eigenvalues of the Hessian [[2a, 0], [0, 2b]]."""
        return np.array([2 * self.a, 2 * self.b])

    @property
    def lr_max(self) -> float:
        """Stable iff lr < 2 / (largest curvature). The steepest direction sets the limit."""
        return 2.0 / self.curvatures.max()

    @property
    def condition_number(self) -> float:
        return float(self.curvatures.max() / self.curvatures.min())

    def path(self, lr: float, steps: int = 30, p0=(2.0, 5.0)) -> Array:
        return gradient_descent(self.grad, p0, lr, steps)


# ---------------------------------------------------------------------------
# 4. Industry example: predicting delivery time (ETA)
# ---------------------------------------------------------------------------
FEATURES = ("distance_km", "prep_min", "traffic")
TRUE_WEIGHTS = np.array([3.2, 0.9, 9.0])   # minutes per km, per prep-minute, per unit traffic
TRUE_BIAS = 4.0                            # fixed handover time, minutes
NOISE_MIN = 3.0                            # real life is noisy


def make_eta_dataset(n: int = 600, seed: int = 7):
    """SYNTHETIC orders that behave like real delivery data. Returns (X, y_minutes)."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.uniform(0.5, 8.0, n),    # distance, km
        rng.uniform(5.0, 25.0, n),   # kitchen prep, minutes
        rng.uniform(0.0, 1.0, n),    # traffic index, 0 = empty roads, 1 = jammed
    ])
    y = X @ TRUE_WEIGHTS + TRUE_BIAS + rng.normal(0.0, NOISE_MIN, n)
    return X, y


def standardize(X: Array):
    """Zero mean, unit variance per feature. Returns (Xs, mean, std)."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    return (X - mu) / sd, mu, sd


def mse_loss(w: Array, b: float, X: Array, y: Array) -> float:
    """L = (1/n) * sum (w.x + b - y)^2   [minutes squared]"""
    r = X @ w + b - y
    return float(np.mean(r ** 2))


def mse_grad(w: Array, b: float, X: Array, y: Array):
    """dL/dw = (2/n) X^T r,   dL/db = (2/n) sum r,   where r = Xw + b - y."""
    r = X @ w + b - y
    n = len(y)
    return (2.0 / n) * (X.T @ r), float((2.0 / n) * r.sum())


def mae(w: Array, b: float, X: Array, y: Array) -> float:
    """Mean absolute error in minutes — the number a product manager cares about."""
    return float(np.mean(np.abs(X @ w + b - y)))


@dataclass
class TrainLog:
    losses: Array        # loss on the full dataset after every update
    maes: Array          # mean abs error (minutes) after every update
    weights: Array       # (updates + 1, n_features)
    biases: Array        # (updates + 1,)


def train_eta(X: Array, y: Array, lr: float = 0.05, epochs: int = 8, batch_size: int = 32,
              seed: int = 0) -> TrainLog:
    """Mini-batch stochastic gradient descent on the linear ETA model.

    batch_size = len(y) gives plain (full-batch) gradient descent.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    losses, maes, ws, bs = [mse_loss(w, b, X, y)], [mae(w, b, X, y)], [w.copy()], [b]
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            gw, gb = mse_grad(w, b, X[idx], y[idx])
            w, b = w - lr * gw, b - lr * gb
            losses.append(mse_loss(w, b, X, y))
            maes.append(mae(w, b, X, y))
            ws.append(w.copy())
            bs.append(b)
    return TrainLog(np.array(losses), np.array(maes), np.array(ws), np.array(bs))


def least_squares(X: Array, y: Array):
    """Closed-form optimum. Used to VERIFY that gradient descent lands in the right place."""
    A = np.column_stack([X, np.ones(len(y))])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return sol[:-1], float(sol[-1])


def mse_lr_max(X: Array) -> float:
    """Full-batch stability limit: lr < 2 / lambda_max of the Hessian (2/n) A^T A."""
    A = np.column_stack([X, np.ones(len(X))])
    H = (2.0 / len(X)) * (A.T @ A)
    return float(2.0 / np.linalg.eigvalsh(H).max())


def mse_condition_number(X: Array) -> float:
    A = np.column_stack([X, np.ones(len(X))])
    ev = np.linalg.eigvalsh((2.0 / len(X)) * (A.T @ A))
    return float(ev.max() / ev.min())


# ---------------------------------------------------------------------------
# 5. Learning-rate schedule used for large models
# ---------------------------------------------------------------------------
def warmup_cosine(step, total_steps: int, peak_lr: float, warmup_steps: int, final_frac: float = 0.1):
    """Linear warmup to peak_lr, then cosine decay to final_frac * peak_lr."""
    step = np.asarray(step, dtype=float)
    warm = peak_lr * step / max(warmup_steps, 1)
    progress = np.clip((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0, 1.0)
    floor = final_frac * peak_lr
    decay = floor + 0.5 * (peak_lr - floor) * (1.0 + np.cos(np.pi * progress))
    return np.where(step < warmup_steps, warm, decay)


# ---------------------------------------------------------------------------
# 6. Facts quoted by the lesson
# ---------------------------------------------------------------------------
def lesson_facts() -> dict:
    """Every number the narration, captions, web page or quiz is allowed to quote.

    The lesson file refers to these by name (``{eta_mae_final}``); the build fills them
    in. A number that is not in this dictionary cannot appear in the lesson.
    """
    slow, brisk, bouncy, wild = (chai_path(lr, 6) for lr in (0.1, 0.4, 0.9, 1.1))
    bowl = Bowl()
    X, y = make_eta_dataset()
    Xs, mu, sd = standardize(X)
    log = train_eta(Xs, y)
    updates_per_epoch = int(np.ceil(len(y) / 32))
    w_raw = log.weights[-1] / sd
    return {
        # chai, one knob
        "sugar_start": SUGAR_START,
        "sugar_best": SUGAR_BEST,
        "chai_grad_start": float(chai_grad(SUGAR_START)),          # -12
        "chai_day1": round(float(slow[1]), 2),                      # 3.2
        "chai_day2": round(float(slow[2]), 2),                      # 4.16
        "chai_day3": round(float(slow[3]), 2),                      # 4.93
        "chai_shrink_percent": round((1 - chai_error_ratio(0.1)) * 100),   # 20
        "chai_brisk_day3": round(float(brisk[3]), 2),               # 7.95
        "chai_bouncy_day1": round(float(bouncy[1]), 1),             # 12.8
        "chai_wild_day1": round(float(wild[1]), 1),                 # 15.2
        "chai_lr_max": CHAI_LR_MAX,
        # chai, two knobs
        "bowl_sensitivity_ratio": round(bowl.condition_number),     # 8
        "bowl_lr_max": bowl.lr_max,                                 # 0.125
        # delivery-time model
        "eta_orders": len(y),
        "eta_batch": 32,
        "eta_epochs": 8,
        "eta_mean_minutes": round(float(y.mean())),
        "eta_mae_start": round(float(log.maes[0]), 1),              # 35.5
        "eta_mae_epoch1": round(float(log.maes[updates_per_epoch]), 1),
        "eta_mae_final": round(float(log.maes[-1]), 1),             # 2.4
        "eta_min_per_km_learned": round(float(w_raw[0]), 1),        # 3.2
        "eta_min_per_km_true": float(TRUE_WEIGHTS[0]),
        "eta_curvature_ratio_raw": int(round(mse_condition_number(X), -2)),   # ~5200
        "eta_curvature_ratio_scaled": round(mse_condition_number(Xs), 1),     # ~1.0
    }


# ---------------------------------------------------------------------------
# 7. The opening story: Meena only hears a few customers a day -> noisy slope
# ---------------------------------------------------------------------------
STORY_LR, STORY_NOISE, STORY_SEED, STORY_DAYS = 0.18, 4.0, 118, 5


def chai_noisy_path(lr: float = STORY_LR, steps: int = STORY_DAYS, noise: float = STORY_NOISE,
                    seed: int = STORY_SEED, s0: float = SUGAR_START):
    """Sugar by day when each day's slope is estimated from a handful of customers.

    Returns (path, felt_slopes). felt_slope = true slope + noise: what Meena perceives.
    This is stochastic gradient descent in one dimension.
    """
    rng = np.random.default_rng(seed)
    s, path, felt = s0, [s0], []
    for _ in range(steps):
        g = float(chai_grad(s)) + rng.normal(0.0, noise)
        felt.append(g)
        s = s - lr * g
        path.append(s)
    return np.array(path), np.array(felt)


def bowl_noisy_path(bowl: "Bowl", lr: float, steps: int, noise: float = 6.0, seed: int = 3, p0=(2.0, 5.0)) -> Array:
    """Two-knob stochastic gradient descent: true gradient + sampling noise each step."""
    rng = np.random.default_rng(seed)
    p = np.asarray(p0, dtype=float)
    path = [p.copy()]
    for _ in range(steps):
        p = p - lr * (bowl.grad(p) + rng.normal(0.0, noise, 2))
        path.append(p.copy())
    return np.array(path)


# ---------------------------------------------------------------------------
# 8. The landscape used throughout the video: f(x, y) = x^2 + 3 y^2
# ---------------------------------------------------------------------------
LANDSCAPE = Bowl(a=1.0, b=3.0, s_best=0.0, t_best=0.0)
LANDSCAPE_START = (-2.6, 1.4)
LANDSCAPE_LR = 0.12


def landscape_path(steps: int = 14, lr: float = LANDSCAPE_LR) -> Array:
    return LANDSCAPE.path(lr=lr, steps=steps, p0=LANDSCAPE_START)
