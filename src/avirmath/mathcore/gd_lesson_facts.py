"""Facts the gradient descent lesson (v2) is allowed to quote. Nothing here is typed by hand."""

from __future__ import annotations

import numpy as np

from avirmath.mathcore import adaptive_filter as af
from avirmath.mathcore import gradient_descent as gd

# learning rates compared in the "how big a step" chapter
LR_CRAWL, LR_GOOD, LR_ZIGZAG, LR_DIVERGE = 0.03, gd.LANDSCAPE_LR, 0.31, 0.35


def lesson_facts() -> dict:
    land = gd.LANDSCAPE
    p0 = np.array(gd.LANDSCAPE_START)
    g0 = land.grad(p0)
    p1 = p0 - gd.LANDSCAPE_LR * g0
    run = af.run_lms()
    before = af.snr_db(run.signal, run.primary)
    after = af.snr_db(run.signal[-1000:], run.output[-1000:])
    w_end = run.weights[-500:].mean(axis=0)
    return {
        "x0": p0[0], "y0": p0[1],
        "f0": round(land.loss(p0), 2),
        "gx0": round(float(g0[0]), 1), "gy0": round(float(g0[1]), 1),
        "lr": gd.LANDSCAPE_LR,
        "x1": round(float(p1[0]), 2), "y1": round(float(p1[1]), 2),
        "f1": round(land.loss(p1), 2),
        "lr_max": round(land.lr_max, 2),                      # 2 / 6
        "mult_x": round(1 - 2 * land.a * gd.LANDSCAPE_LR, 2),
        "mult_y": round(1 - 2 * land.b * gd.LANDSCAPE_LR, 2),
        "lr_crawl": LR_CRAWL, "lr_zigzag": LR_ZIGZAG, "lr_diverge": LR_DIVERGE,
        "lms_samples": af.N_SAMPLES,
        "lms_gain_db": int(round(after - before)),
        "lms_w0": round(float(w_end[0]), 1), "lms_w1": round(float(w_end[1]), 1),
        "leak0": float(af.LEAK[0]), "leak1": float(af.LEAK[1]),
    }
