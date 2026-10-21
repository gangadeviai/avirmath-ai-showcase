"""Widrow-Hoff LMS adaptive noise canceller, and a rugged landscape for the "today" chapter.

Setup (the classic adaptive noise-cancelling arrangement):
    primary microphone   d[t] = s[t] + v[t]        wanted signal + noise that leaked in
    reference microphone x[t]                      the noise source alone
    the leak             v[t] = h0 x[t] + h1 x[t-1]   (unknown to the filter)
The filter predicts the leak from the reference, y = w0 x[t] + w1 x[t-1], and outputs
e = d - y. LMS nudges w after EVERY sample:  w <- w + 2 mu e [x[t], x[t-1]].
When w reaches h, the output e is the clean signal.

With white, unit-variance noise the mean squared error is exactly
    J(w) = var(s) + (w0 - h0)^2 + (w1 - h1)^2
a perfectly round bowl whose bottom is the true leak h. LMS is stochastic gradient descent on J.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray

LEAK = np.array([0.8, -0.5])
MU = 0.004
N_SAMPLES = 4000


def wanted_signal(n: int = N_SAMPLES) -> Array:
    t = np.arange(n)
    return 0.6 * np.sin(2 * np.pi * 0.011 * t) + 0.3 * np.sin(2 * np.pi * 0.027 * t + 0.5)


@dataclass
class LmsRun:
    signal: Array      # s[t]   what we want
    primary: Array     # d[t]   what the microphone hears
    output: Array      # e[t]   what the filter delivers
    weights: Array     # (n + 1, 2) weight vector after each sample


def run_lms(n: int = N_SAMPLES, mu: float = MU, seed: int = 11) -> LmsRun:
    rng = np.random.default_rng(seed)
    s = wanted_signal(n)
    x = rng.normal(0.0, 1.0, n)
    x_prev = np.concatenate([[0.0], x[:-1]])
    d = s + LEAK[0] * x + LEAK[1] * x_prev
    w = np.zeros(2)
    ws, out = [w.copy()], np.zeros(n)
    for t in range(n):
        u = np.array([x[t], x_prev[t]])
        e = d[t] - w @ u
        w = w + 2 * mu * e * u            # one-sample estimate of the gradient of J
        out[t] = e
        ws.append(w.copy())
    return LmsRun(s, d, out, np.array(ws))


def error_surface(w0, w1, signal_power: float) -> Array:
    """Exact mean squared error J(w) for white unit-variance reference noise."""
    return signal_power + (np.asarray(w0) - LEAK[0]) ** 2 + (np.asarray(w1) - LEAK[1]) ** 2


def snr_db(clean: Array, observed: Array) -> float:
    noise = observed - clean
    return float(10 * np.log10(np.mean(clean ** 2) / np.mean(noise ** 2)))


# ---------------------------------------------------------------------------
# A rugged, non-convex landscape: a cartoon of a neural network's loss
# ---------------------------------------------------------------------------
def rugged(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return 0.30 * (x ** 2 + y ** 2) + 0.9 * np.sin(1.7 * x) * np.cos(1.9 * y) + 1.0


def rugged_grad(p) -> Array:
    x, y = np.asarray(p, dtype=float)
    return np.array([0.60 * x + 0.9 * 1.7 * np.cos(1.7 * x) * np.cos(1.9 * y),
                     0.60 * y - 0.9 * 1.9 * np.sin(1.7 * x) * np.sin(1.9 * y)])


def rugged_path(p0=(2.5, 2.1), lr: float = 0.09, steps: int = 45) -> Array:
    p = np.asarray(p0, dtype=float)
    path = [p.copy()]
    for _ in range(steps):
        p = p - lr * rugged_grad(p)
        path.append(p.copy())
    return np.array(path)
