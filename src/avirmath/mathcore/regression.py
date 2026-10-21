"""Linear and logistic regression — the single source of mathematical truth for the lesson.

Sections
    1. Least squares (Legendre 1805, Gauss 1809): the line, the normal equations, the closed form
    2. The same answer by gradient descent (lesson 1), and the 3-D plane for two inputs
    3. Logistic regression: the sigmoid (Verhulst 1838), log-odds, the likelihood, the gradient (p - y) X
    4. Industry: credit scoring (logistic), demand forecasting (linear), medical risk (logistic)
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray

# Worked data: five points, by hand
X1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
Y1 = np.array([2.0, 2.8, 4.1, 4.9, 6.2])


# ---------------------------------------------------------------------------
# 1. Least squares
# ---------------------------------------------------------------------------
def fit_line(x, y) -> tuple[float, float]:
    """Slope and intercept from the closed form: w = cov(x,y)/var(x), b = ybar - w xbar."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    xm, ym = x.mean(), y.mean()
    w = float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))
    return w, float(ym - w * xm)


def sse(w: float, b: float, x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.sum((w * x + b - y) ** 2))


def sse_grad(w: float, b: float, x, y) -> tuple[float, float]:
    """d/dw, d/db of the sum of squared residuals."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    r = w * x + b - y
    return float(2 * np.sum(r * x)), float(2 * np.sum(r))


def normal_equations(X, y) -> Array:
    """theta = (X^T X)^{-1} X^T y with X carrying a column of ones."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    return np.linalg.solve(X.T @ X, X.T @ y)


def design(x) -> Array:
    x = np.asarray(x, dtype=float)
    return np.column_stack([x, np.ones_like(x)])


def gd_line(x, y, lr: float = 0.01, steps: int = 400, w0: float = 0.0, b0: float = 0.0) -> Array:
    """Gradient descent on the SSE. Returns (steps+1, 3): w, b, loss."""
    w, b = w0, b0
    rows = []
    for _ in range(steps + 1):
        rows.append((w, b, sse(w, b, x, y)))
        gw, gb = sse_grad(w, b, x, y)
        w, b = w - lr * gw, b - lr * gb
    return np.array(rows)


def make_plane_data(n: int = 120, seed: int = 2):
    """SYNTHETIC two-input data: y = 1.5 x1 - 0.8 x2 + 2 + noise, for the 3-D fitted plane."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, 2))
    y = 1.5 * X[:, 0] - 0.8 * X[:, 1] + 2.0 + 0.4 * rng.normal(size=n)
    return X, y


# ---------------------------------------------------------------------------
# 3. Logistic regression
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1 - p))


def nll(w, b, X, y) -> float:
    """Negative log-likelihood (cross-entropy) of labels y in {0,1}."""
    p = sigmoid(np.asarray(X) @ np.asarray(w) + b)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def nll_grad(w, b, X, y):
    """d/dw = X^T (p - y),  d/db = sum (p - y)  — the same 'prediction minus truth' as softmax."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    p = sigmoid(X @ np.asarray(w) + b)
    return X.T @ (p - y), float(np.sum(p - y))


def fit_logistic(X, y, lr: float = 0.05, steps: int = 300):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    w, b = np.zeros(X.shape[1]), 0.0
    hist = []
    for _ in range(steps + 1):
        hist.append((w.copy(), b, nll(w, b, X, y)))
        gw, gb = nll_grad(w, b, X, y)
        w, b = w - lr * gw, b - lr * gb
    return w, b, hist


def accuracy(w, b, X, y) -> float:
    return float(np.mean((sigmoid(np.asarray(X) @ w + b) > 0.5) == (np.asarray(y) > 0.5)))


# The hand-worked logistic example: hours studied -> passed (1) or not (0)
HOURS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
PASSED = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])


def make_credit_data(n: int = 800, seed: int = 11):
    """SYNTHETIC credit applications: income (k), debt ratio, late payments -> default (1) or repaid (0).

    Shaped like the public record of such models: default risk falls with income, rises with debt
    ratio and with past late payments. Used only to show the mathematics on a realistic problem.
    """
    rng = np.random.default_rng(seed)
    income = rng.uniform(20, 120, n)
    debt = rng.uniform(0.05, 0.8, n)
    late = rng.poisson(0.8, n)
    z = -1.0 - 0.03 * (income - 60) + 4.0 * (debt - 0.4) + 0.8 * late
    y = (rng.random(n) < sigmoid(z)).astype(float)
    X = np.column_stack([income, debt, late])
    Xs = (X - X.mean(0)) / X.std(0)
    return Xs, y, X


def odds_ratio(coef: float) -> float:
    """exp(coef): how the odds multiply per unit (standardised) increase of a feature."""
    return float(np.exp(coef))


# ---------------------------------------------------------------------------
# 5. Real datasets (lesson v3): German Credit (logistic) and hourly bike rentals (linear)
#    Both from the UCI Machine Learning Repository, CC BY 4.0, stored under <repo>/data.
# ---------------------------------------------------------------------------
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "data"
SEED = 0

# German Credit: the eight attributes this lesson uses (names as in the UCI documentation)
GERMAN_FEATURES = ["checking account", "duration (months)", "credit history", "credit amount (DM)",
                   "savings", "employment (years)", "instalment rate (% income)", "age (years)"]
_ORDINAL = {"A11": 0, "A12": 1, "A13": 2, "A14": 3,            # checking: <0 DM, 0-200, >=200, none
            "A30": 0, "A31": 1, "A32": 2, "A33": 3, "A34": 4,  # history: no credits ... critical account
            "A61": 0, "A62": 1, "A63": 2, "A64": 3, "A65": 4,  # savings: <100 ... unknown/none
            "A71": 0, "A72": 1, "A73": 2, "A74": 3, "A75": 4}  # employment: unemployed ... >=7 years


def load_german():
    """1,000 real loans. Returns raw feature table (n, 8), y (1 = bad loan), and the row-wise codes."""
    rows = [ln.split() for ln in (DATA / "german" / "german.data").read_text().splitlines() if ln.strip()]
    X = np.array([[_ORDINAL[r[0]], float(r[1]), _ORDINAL[r[2]], float(r[4]), _ORDINAL[r[5]], _ORDINAL[r[6]], float(r[7]), float(r[12])] for r in rows])
    y = np.array([1.0 if r[20] == "2" else 0.0 for r in rows])
    return X, y


def split(X, y, n_train: int, seed: int = SEED):
    """Shuffle once with a fixed seed, then hold out everything after n_train for testing."""
    idx = np.random.default_rng(seed).permutation(len(y))
    tr, te = idx[:n_train], idx[n_train:]
    return X[tr], y[tr], X[te], y[te]


def standardise(X_train, X_other=None):
    mu, sd = X_train.mean(0), X_train.std(0)
    f = lambda A: (A - mu) / sd
    return (f(X_train), f(X_other), mu, sd) if X_other is not None else (f(X_train), mu, sd)


def auc(scores, y) -> float:
    """Area under the ROC curve by the rank (Mann-Whitney) formula: P(score of a positive > score of a negative)."""
    scores, y = np.asarray(scores, dtype=float), np.asarray(y) > 0.5
    pos, neg = scores[y], scores[~y]
    gt = (pos[:, None] > neg[None, :]).mean(); eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error: |confidence - observed frequency| averaged over probability bins."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1); err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any(): err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def confusion(p, y, threshold: float) -> dict:
    pred = np.asarray(p) >= threshold; y = np.asarray(y) > 0.5
    tp, fp, fn, tn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum()), int((~pred & ~y).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "recall_bad": tp / max(tp + fn, 1), "flagged": int(pred.sum())}


def credit_model(n_train: int = 800, steps: int = 3000, lr: float = 0.0015):
    """Logistic regression on the real German Credit data, trained by the gradient derived in the lesson."""
    X, y = load_german()
    Xtr, ytr, Xte, yte = split(X, y, n_train)
    Ztr, Zte, mu, sd = standardise(Xtr, Xte)
    w, b, hist = fit_logistic(Ztr, ytr, lr=lr, steps=steps)
    p_te = sigmoid(Zte @ w + b)
    return {"w": w, "b": b, "hist": hist, "mu": mu, "sd": sd, "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte, "Zte": Zte, "p_te": p_te}


# Bike sharing (hour.csv): features and the target count of rentals
BIKE_FEATURES = ["temperature (°C)", "humidity (%)", "wind speed (km/h)", "working day", "hour (sin)", "hour (cos)"]


def load_bike():
    """17,379 real hours of rentals. Returns feature table (n, 6), y = rentals, plus hour, casual, registered."""
    lines = (DATA / "bike" / "hour.csv").read_text().splitlines()
    head = lines[0].split(","); col = {h: i for i, h in enumerate(head)}
    rows = [ln.split(",") for ln in lines[1:] if ln.strip()]
    g = lambda name: np.array([float(r[col[name]]) for r in rows])
    hr = g("hr")
    X = np.column_stack([g("temp") * 41.0, g("hum") * 100.0, g("windspeed") * 67.0, g("workingday"),
                         np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)])
    return X, g("cnt"), hr, g("casual"), g("registered")


def linear_fit(X, y):
    """Normal equations with an intercept column appended last."""
    A = np.column_stack([X, np.ones(len(y))])
    return normal_equations(A, y)


def linear_predict(theta, X):
    return np.column_stack([X, np.ones(len(X))]) @ theta


def rmse(a, b) -> float: return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
def mae(a, b) -> float: return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
def r2(y, yhat) -> float: y = np.asarray(y); return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def bike_model(train_frac: float = 0.8):
    """Linear regression (normal equations) on the real hourly rental data, with the leakage and extrapolation demos."""
    X, y, hr, casual, registered = load_bike()
    n_train = int(len(y) * train_frac)
    Xtr, ytr, Xte, yte = split(X, y, n_train)
    theta = linear_fit(Xtr, ytr)
    yhat = linear_predict(theta, Xte)
    # residual pattern by hour of day (why a straight line cannot capture rush hour)
    hr_te = split(hr.reshape(-1, 1), y, n_train)[2].ravel()
    resid_by_hour = np.array([float(np.mean((yte - yhat)[hr_te == h])) for h in range(24)])
    # leakage: cnt = casual + registered, so adding them "predicts" perfectly
    leak = np.column_stack([X, casual, registered])
    Ltr, _, Lte, _ = split(leak, y, n_train)
    theta_leak = linear_fit(Ltr, ytr)
    leak_rmse = rmse(linear_predict(theta_leak, Lte), yte)
    # extrapolation: ask for a temperature far outside the data
    t_max = float(X[:, 0].max())
    x_extrap = np.array([[60.0, 50.0, 10.0, 1.0, np.sin(2 * np.pi * 17 / 24), np.cos(2 * np.pi * 17 / 24)]])
    x_in = x_extrap.copy(); x_in[0, 0] = 30.0
    return {"theta": theta, "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte, "yhat": yhat, "hr_te": hr_te,
            "rmse": rmse(yhat, yte), "mae": mae(yhat, yte), "r2": r2(yte, yhat), "resid_by_hour": resid_by_hour,
            "leak_rmse": leak_rmse, "t_max": t_max, "extrap_pred": float(linear_predict(theta, x_extrap)[0]),
            "in_range_pred": float(linear_predict(theta, x_in)[0]), "n": len(y), "n_train": n_train}


def _real_facts() -> dict:
    c = credit_model(); b = bike_model()
    yte, p = c["yte"], c["p_te"]
    c05, c03 = confusion(p, yte, 0.5), confusion(p, yte, 0.3)
    base = 1 - float(c["ytr"].mean())                         # "approve everyone" accuracy on a 70/30 split
    # an example applicant from the test set whose risk sits near the decision boundary
    k = int(np.argmin(np.abs(p - 0.55)))
    ex = c["Xte"][k]
    ors = {name: round(odds_ratio(wi), 2) for name, wi in zip(GERMAN_FEATURES, c["w"])}
    th = b["theta"]
    return {
        "german_n": 1000, "german_bad_pct": 30, "german_train": 800, "german_test": 200, "german_features": len(GERMAN_FEATURES),
        "german_acc": round(float(np.mean((p >= 0.5) == (yte > 0.5))) * 100), "german_base_acc": round(base * 100),
        "german_auc": round(auc(p, yte), 2), "german_ece": round(ece(p, yte), 3),
        "german_nll_start": round(c["hist"][0][2] / len(c["ytr"]), 3), "german_nll_end": round(c["hist"][-1][2] / len(c["ytr"]), 3),
        "german_recall_05": round(c05["recall_bad"] * 100), "german_recall_03": round(c03["recall_bad"] * 100),
        "german_flagged_05": c05["flagged"], "german_flagged_03": c03["flagged"], "german_test_bad": int(yte.sum()),
        "or_checking": ors["checking account"], "or_duration": ors["duration (months)"], "or_history": ors["credit history"],
        "or_amount": ors["credit amount (DM)"], "or_savings": ors["savings"], "or_employment": ors["employment (years)"],
        "or_instalment": ors["instalment rate (% income)"], "or_age": ors["age (years)"],
        "ex_p": round(float(p[k]) * 100), "ex_duration": int(ex[1]), "ex_amount": int(ex[3]), "ex_age": int(ex[7]),
        "ex_checking": ["below zero", "under 200 marks", "200 marks or more", "no account"][int(ex[0])],
        "bike_n": b["n"], "bike_train": b["n_train"], "bike_test": b["n"] - b["n_train"], "bike_mean": round(float(b["ytr"].mean())),
        "bike_rmse": round(b["rmse"]), "bike_mae": round(b["mae"]), "bike_r2": round(b["r2"], 2),
        "bike_coef_temp": round(float(th[0]), 1), "bike_coef_hum": round(float(th[1]), 1), "bike_coef_work": round(float(th[3])),
        "bike_leak_rmse": round(b["leak_rmse"], 6), "bike_t_max": round(b["t_max"]), "bike_extrap_pred": round(b["extrap_pred"]),
        "bike_in_range_pred": round(b["in_range_pred"]), "bike_resid_8": round(float(b["resid_by_hour"][8])), "bike_resid_17": round(float(b["resid_by_hour"][17])),
        "bike_resid_3": round(float(b["resid_by_hour"][3])), "bike_over_3": round(abs(float(b["resid_by_hour"][3]))),
    }


# ---------------------------------------------------------------------------
# Facts the lesson may quote
# ---------------------------------------------------------------------------
def lesson_facts() -> dict:
    w, b = fit_line(X1, Y1)
    traj = gd_line(X1, Y1)
    xm, ym = X1.mean(), Y1.mean()
    sxy, sxx = float(np.sum((X1 - xm) * (Y1 - ym))), float(np.sum((X1 - xm) ** 2))
    Xp, yp = make_plane_data()
    theta = normal_equations(np.column_stack([Xp, np.ones(len(yp))]), yp)
    wl, bl, hist = fit_logistic(HOURS.reshape(-1, 1), PASSED, lr=0.1, steps=2000)
    Xc, yc, raw = make_credit_data()
    wc, bc, hc = fit_logistic(Xc, yc, lr=0.002, steps=400)
    return {
        "x": X1.tolist(), "y": Y1.tolist(), "xbar": float(xm), "ybar": float(ym), "sxy": round(sxy, 2), "sxx": round(sxx, 2),
        "w": round(w, 3), "b": round(b, 3), "sse_fit": round(sse(w, b, X1, Y1), 3), "sse_zero": round(sse(0, 0, X1, Y1), 2),
        "gd_w": round(float(traj[-1, 0]), 3), "gd_b": round(float(traj[-1, 1]), 3), "gd_steps": len(traj) - 1,
        "plane": [round(float(t), 2) for t in theta],
        "log_w": round(float(wl[0]), 2), "log_b": round(bl, 2), "log_boundary_hours": round(-bl / float(wl[0]), 2),
        "log_p_2h": round(float(sigmoid(wl[0] * 2 + bl)), 2), "log_p_4h": round(float(sigmoid(wl[0] * 4 + bl)), 2),
        "credit_n": len(yc), "credit_default_pct": round(float(yc.mean()) * 100), "credit_acc": round(accuracy(wc, bc, Xc, yc) * 100),
        "credit_nll_start": round(hc[0][2] / len(yc), 3), "credit_nll_end": round(hc[-1][2] / len(yc), 3),
        "credit_or_income": round(odds_ratio(wc[0]), 2), "credit_or_debt": round(odds_ratio(wc[1]), 2), "credit_or_late": round(odds_ratio(wc[2]), 2),
        **_real_facts(),
    }
