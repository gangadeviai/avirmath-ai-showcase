"""Softmax and cross-entropy — the single source of mathematical truth for the lesson.

Sections
    1. Softmax: scores -> shares of 100%; temperature; shift invariance; the max-subtraction trick
    2. Why exponentials: Boltzmann (1868) and the Luce choice axiom; log-odds / sigmoid as the 2-class case
    3. Cross-entropy: surprise, entropy, KL; the loss of a classifier; its gradient is (p - y)
    4. Where AI uses it: the classifier head, next-token prediction, temperature sampling, attention weights
    5. Industry: ImageNet-scale classification, LLM decoding temperature, medical triage calibration
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray

SCORES = np.array([2.0, 1.0, 0.1])            # the worked example: three classes
LABELS = ("cat", "dog", "fox")
TARGET = 0                                     # the true class in the worked example


# ---------------------------------------------------------------------------
# 1. Softmax
# ---------------------------------------------------------------------------
def softmax(z, temperature: float = 1.0) -> Array:
    z = np.asarray(z, dtype=float) / temperature
    z = z - z.max()                            # shift invariance: changes nothing, prevents overflow
    e = np.exp(z)
    return e / e.sum()


def softmax_naive(z) -> Array:
    """Textbook formula without the shift (overflows for large scores)."""
    e = np.exp(np.asarray(z, dtype=float))
    return e / e.sum()


def softmax_jacobian(p) -> Array:
    """d p_i / d z_j = p_i (delta_ij - p_j)."""
    p = np.asarray(p, dtype=float)
    return np.diag(p) - np.outer(p, p)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def two_class_softmax_equals_sigmoid(z1: float, z2: float) -> tuple[float, float]:
    return float(softmax([z1, z2])[0]), float(sigmoid(z1 - z2))


# ---------------------------------------------------------------------------
# 2. Boltzmann
# ---------------------------------------------------------------------------
def boltzmann(energies, kT: float = 1.0) -> Array:
    """p_i proportional to exp(-E_i / kT): softmax of -E/kT."""
    return softmax(-np.asarray(energies, dtype=float), temperature=kT)


# ---------------------------------------------------------------------------
# 3. Cross-entropy
# ---------------------------------------------------------------------------
def surprise(p: float) -> float:
    return float(-np.log(p))


def entropy(p) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def cross_entropy(y, q) -> float:
    """H(y, q) = -sum y log q. For a one-hot y this is -log q[target]."""
    y, q = np.asarray(y, dtype=float), np.asarray(q, dtype=float)
    return float(-np.sum(y * np.log(q)))


def kl(p, q) -> float:
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def one_hot(k: int, n: int) -> Array:
    y = np.zeros(n); y[k] = 1.0
    return y


def loss_and_grad(z, target: int):
    """Loss = -log softmax(z)[target];  dLoss/dz = p - y  (the famous one-liner)."""
    p = softmax(z)
    y = one_hot(target, len(p))
    return cross_entropy(y, p), p - y


def numerical_grad(z, target: int, eps: float = 1e-6) -> Array:
    z = np.asarray(z, dtype=float)
    g = np.zeros_like(z)
    for i in range(z.size):
        d = np.zeros_like(z); d[i] = eps
        g[i] = (loss_and_grad(z + d, target)[0] - loss_and_grad(z - d, target)[0]) / (2 * eps)
    return g


def train_scores(z0, target: int, lr: float = 1.0, steps: int = 8):
    """Gradient descent directly on the scores: shows the loss falling as p[target] rises."""
    z = np.asarray(z0, dtype=float).copy()
    rows = []
    for _ in range(steps + 1):
        L, g = loss_and_grad(z, target)
        rows.append((z.copy(), softmax(z), L))
        z = z - lr * g
    return rows


# ---------------------------------------------------------------------------
# 4-5. Where AI and industry use it
# ---------------------------------------------------------------------------
NEXT_TOKEN_LOGITS = {"the": 6.1, "a": 5.4, "an": 3.0, "my": 2.6, "this": 2.2, "zebra": -1.0}


def next_token_distribution(temperature: float = 1.0) -> dict:
    words = list(NEXT_TOKEN_LOGITS)
    p = softmax(list(NEXT_TOKEN_LOGITS.values()), temperature)
    return dict(zip(words, p))


def expected_calibration(probs, correct) -> float:
    """Mean |confidence - accuracy| over 5 confidence bins (a calibration check used in medical triage models)."""
    probs, correct = np.asarray(probs, dtype=float), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, 6)
    err, n = 0.0, len(probs)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi)
        if m.any():
            err += m.sum() / n * abs(probs[m].mean() - correct[m].mean())
    return float(err)


def make_classifier_demo(n: int = 600, seed: int = 4):
    """SYNTHETIC 3-class problem in 2-D solved by a linear layer + softmax, trained by gradient descent."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 2.0], [-1.8, -1.0], [1.8, -1.0]])
    labels = rng.integers(0, 3, n)
    X = centres[labels] + 0.8 * rng.normal(size=(n, 2))
    Y = np.eye(3)[labels]
    W, b = np.zeros((3, 2)), np.zeros(3)
    losses, accs = [], []
    for _ in range(60):
        Z = X @ W.T + b
        P = np.apply_along_axis(softmax, 1, Z)
        losses.append(float(-np.mean(np.sum(Y * np.log(P), axis=1))))
        accs.append(float(np.mean(P.argmax(1) == labels)))
        G = (P - Y) / n                          # dLoss/dZ, averaged
        W -= 0.5 * G.T @ X
        b -= 0.5 * G.sum(0)
    return X, labels, W, b, np.array(losses), np.array(accs)


# ---------------------------------------------------------------------------
# 6. Inside real models: lung-cancer screening and language models
# ---------------------------------------------------------------------------
# Published reference numbers (sources listed in lessons/softmax-cross-entropy/lesson.yaml).
ARDILA_2019 = {"auc_pct": 94.4, "cases": 6716, "validation_cases": 1139, "fp_reduction_pct": 11, "fn_reduction_pct": 5, "radiologists": 6}
GPT2_VOCAB = 50257
PPL_EXAMPLE_CE = 3.0


def perplexity(cross_entropy_nats: float) -> float:
    """Perplexity = e^(cross-entropy): the size of the uniform guess the model is as unsure as."""
    return float(np.exp(cross_entropy_nats))


def _nodule_features(diam_mm, spic):
    return np.column_stack([(np.asarray(diam_mm, dtype=float) - 10.0) / 5.0, (np.asarray(spic, dtype=float) - 0.5) * 2.0])


def make_nodule_demo(n: int = 400, seed: int = 11):
    """SYNTHETIC lung-nodule risk model: two features (diameter in mm, spiculation 0-1), two classes
    (benign, malignant), a linear layer + softmax trained by gradient descent on cross-entropy.
    Not a medical model: it exists to show where softmax and cross-entropy sit in a screening classifier."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    diam = np.where(labels == 1, rng.normal(14, 4, n), rng.normal(6, 2.5, n)).clip(2, 30)
    spic = np.where(labels == 1, rng.beta(4, 2, n), rng.beta(2, 5, n))
    X = _nodule_features(diam, spic)
    Y = np.eye(2)[labels]
    W, b = np.zeros((2, 2)), np.zeros(2)
    losses, accs = [], []
    for _ in range(80):
        Z = X @ W.T + b
        P = np.apply_along_axis(softmax, 1, Z)
        losses.append(float(-np.mean(np.sum(Y * np.log(P), axis=1))))
        accs.append(float(np.mean(P.argmax(1) == labels)))
        G = (P - Y) / n
        W -= 0.8 * G.T @ X
        b -= 0.8 * G.sum(0)
    return X, labels, diam, spic, W, b, np.array(losses), np.array(accs)


def nodule_risk(diam_mm: float, spic: float, W, b) -> float:
    """Probability of 'malignant' from the synthetic model: softmax over two scores = sigmoid of their difference."""
    x = _nodule_features([diam_mm], [spic])[0]
    return float(softmax(W @ x + b)[1])


# ---------------------------------------------------------------------------
# 7. A real dataset (lesson v3): UCI Dermatology — six erythemato-squamous diseases
#    H. A. Guvenir, G. Demiroz, N. Ilter, Artificial Intelligence in Medicine 13(3), 1998. CC BY 4.0.
# ---------------------------------------------------------------------------
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "data"
DERM_CLASSES = ["psoriasis", "seboreic dermatitis", "lichen planus", "pityriasis rosea", "chronic dermatitis", "pityriasis rubra pilaris"]
DERM_CLINICAL = 11          # attributes 1-11 are clinical (plus age, attribute 34); 12-33 are histopathological (biopsy)
DERM_SEED = 0


def load_dermatology():
    """366 real patients: 34 attributes (33 graded 0-3, plus age; 8 ages missing -> median), diagnosis 0..5."""
    rows = [ln.split(",") for ln in (DATA / "dermatology" / "dermatology.data").read_text().splitlines() if ln.strip()]
    X = np.array([[float(v) if v != "?" else np.nan for v in r[:34]] for r in rows])
    y = np.array([int(r[34]) - 1 for r in rows])
    X[np.isnan(X[:, 33]), 33] = np.nanmedian(X[:, 33])
    return X, y


def split(X, y, n_train: int, seed: int = DERM_SEED):
    idx = np.random.default_rng(seed).permutation(len(y))
    tr, te = idx[:n_train], idx[n_train:]
    return X[tr], y[tr], X[te], y[te]


def standardise(X_train, X_other):
    mu, sd = X_train.mean(0), X_train.std(0); sd[sd == 0] = 1.0
    return (X_train - mu) / sd, (X_other - mu) / sd, mu, sd


def softmax_rows(Z):
    return np.apply_along_axis(softmax, 1, Z)


def fit_softmax(X, y, k: int, lr: float = 0.5, steps: int = 1500, l2: float = 1e-3):
    """Multinomial logistic regression: W (k x d), b (k), trained by gradient descent on mean cross-entropy
    with a small L2 penalty so that a perfectly separable training set does not drive the weights to infinity."""
    X = np.asarray(X, dtype=float); n, d = X.shape
    Y = np.eye(k)[y]
    W, b = np.zeros((k, d)), np.zeros(k)
    hist = []
    for _ in range(steps + 1):
        P = softmax_rows(X @ W.T + b)
        loss = float(-np.mean(np.sum(Y * np.log(np.clip(P, 1e-12, 1)), axis=1)))
        hist.append((loss, float(np.mean(P.argmax(1) == y))))
        G = (P - Y) / n                              # dLoss/dZ: prediction minus truth, averaged
        W -= lr * (G.T @ X + l2 * W)
        b -= lr * G.sum(0)
    return W, b, hist


def predict_proba(W, b, X, temperature: float = 1.0):
    return softmax_rows((np.asarray(X) @ W.T + b) / temperature)


def confusion_matrix(y_true, y_pred, k: int):
    M = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred): M[t, p] += 1
    return M


def ece_multiclass(P, y, bins: int = 10) -> float:
    """Expected calibration error on the top-class confidence."""
    conf, pred = P.max(1), P.argmax(1); correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): err += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(err)


def fit_temperature(Z, y, grid=np.linspace(0.5, 5.0, 91)) -> float:
    """Temperature scaling (Guo et al., ICML 2017): one scalar that minimises cross-entropy of the held-out logits."""
    best, bestT = np.inf, 1.0
    for T in grid:
        P = softmax_rows(Z / T); nll = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))
        if nll < best: best, bestT = nll, float(T)
    return bestT


from functools import lru_cache


@lru_cache(maxsize=4)
def dermatology_model(train_frac: float = 0.8, noise_frac: float = 0.2, seed: int = DERM_SEED):
    """Softmax regression on the real dermatology data, plus the failure demonstrations the lesson shows."""
    X, y = load_dermatology(); k = len(DERM_CLASSES)
    n_train = int(round(len(y) * train_frac))
    Xtr, ytr, Xte, yte = split(X, y, n_train, seed)
    Ztr, Zte, mu, sd = standardise(Xtr, Xte)
    W, b, hist = fit_softmax(Ztr, ytr, k)
    logits = Zte @ W.T + b
    P = softmax_rows(logits); pred = P.argmax(1)
    cm = confusion_matrix(yte, pred, k)
    recall = np.array([cm[i, i] / max(cm[i].sum(), 1) for i in range(k)])
    top2 = np.mean([yt in np.argsort(-p)[:2] for p, yt in zip(P, yte)])
    T = fit_temperature(logits, yte)
    P_T = softmax_rows(logits / T)
    # label noise: flip a fraction of the training labels at random and retrain
    rng = np.random.default_rng(seed + 1)
    flip = rng.random(n_train) < noise_frac
    y_noisy = ytr.copy(); y_noisy[flip] = (y_noisy[flip] + rng.integers(1, k, flip.sum())) % k
    Wn, bn, _ = fit_softmax(Ztr, y_noisy, k)
    acc_noisy = float(np.mean(predict_proba(Wn, bn, Zte).argmax(1) == yte))
    # over-confidence: the same model trained without regularisation for much longer on a separable set
    Wo, bo, _ = fit_softmax(Ztr, ytr, k, lr=0.5, steps=6000, l2=0.0)
    Po = predict_proba(Wo, bo, Zte)
    acc_over, conf_over, ece_over = float(np.mean(Po.argmax(1) == yte)), float(Po.max(1).mean()), ece_multiclass(Po, yte)
    # distribution shift in the clinic: the biopsy is not available, so only the 12 clinical findings + age are known
    clin = list(range(DERM_CLINICAL)) + [33]
    Wc, bc, _ = fit_softmax(Ztr[:, clin], ytr, k)
    acc_clinical = float(np.mean(predict_proba(Wc, bc, Zte[:, clin]).argmax(1) == yte))
    # the "close second": test patients whose top two probabilities are within 0.2 of each other
    srt = -np.sort(-P, axis=1); close = (srt[:, 0] - srt[:, 1]) < 0.2
    # a real patient to show: the closest call that the model still gets right
    cand = np.where(close & (pred == yte))[0]
    kx = int(cand[np.argmin((srt[cand, 0] - srt[cand, 1]))]) if len(cand) else int(np.argmin(srt[:, 0] - srt[:, 1]))
    return {"W": W, "b": b, "hist": hist, "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte, "P": P, "pred": pred, "cm": cm, "recall": recall,
            "acc": float(np.mean(pred == yte)), "top2": float(top2), "ece": ece_multiclass(P, yte), "T": T, "ece_T": ece_multiclass(P_T, yte),
            "conf_mean": float(P.max(1).mean()), "conf_mean_T": float(P_T.max(1).mean()), "acc_noisy": acc_noisy, "n_close": int(close.sum()),
            "acc_over": acc_over, "conf_over": conf_over, "ece_over": ece_over, "acc_clinical": acc_clinical,
            "example": kx, "n": len(y), "n_train": n_train, "n_test": len(yte), "class_counts": np.bincount(y, minlength=k)}


def _derm_facts() -> dict:
    D = dermatology_model(); k = len(DERM_CLASSES)
    ex = D["example"]; p = D["P"][ex]; order = np.argsort(-p)
    rare = int(np.argmin(D["class_counts"]))
    return {
        "derm_n": D["n"], "derm_train": D["n_train"], "derm_test": D["n_test"], "derm_features": 34, "derm_clinical": DERM_CLINICAL + 1, "derm_histo": 33 - DERM_CLINICAL,
        "derm_classes": k, "derm_counts": [int(c) for c in D["class_counts"]], "derm_rare": DERM_CLASSES[rare], "derm_rare_n": int(D["class_counts"][rare]),
        "derm_acc": round(D["acc"] * 100), "derm_top2": round(D["top2"] * 100), "derm_correct": int(round(D["acc"] * D["n_test"])),
        "derm_loss_start": round(D["hist"][0][0], 3), "derm_loss_end": round(D["hist"][-1][0], 3),
        "derm_recall_rare": round(float(D["recall"][rare]) * 100), "derm_recall_min": round(float(D["recall"].min()) * 100),
        "derm_ece": round(D["ece"], 3), "derm_T": round(D["T"], 2), "derm_ece_T": round(D["ece_T"], 3), "derm_conf": round(D["conf_mean"] * 100), "derm_conf_T": round(D["conf_mean_T"] * 100),
        "derm_acc_over": round(D["acc_over"] * 100), "derm_conf_over": round(D["conf_over"] * 100, 1), "derm_ece_over": round(D["ece_over"], 3), "derm_acc_clinical": round(D["acc_clinical"] * 100),
        "derm_acc_noisy": round(D["acc_noisy"] * 100), "derm_noise_pct": 20, "derm_n_close": D["n_close"],
        "ex_top": DERM_CLASSES[order[0]], "ex_top_pct": round(float(p[order[0]]) * 100), "ex_second": DERM_CLASSES[order[1]], "ex_second_pct": round(float(p[order[1]]) * 100),
        "ex_truth": DERM_CLASSES[int(D["yte"][ex])], "ex_age": int(D["Xte"][ex, 33]),
        "paper_acc": 96.2,
    }


# ---------------------------------------------------------------------------
# Facts the lesson may quote
# ---------------------------------------------------------------------------
def lesson_facts() -> dict:
    p = softmax(SCORES)
    e = np.exp(SCORES)
    L, g = loss_and_grad(SCORES, TARGET)
    cold, hot = softmax(SCORES, 0.3), softmax(SCORES, 5.0)
    rows = train_scores(SCORES, TARGET)
    nt1, nt3 = next_token_distribution(1.0), next_token_distribution(0.3)
    X, labels, W, b, losses, accs = make_classifier_demo()
    return {
        "z": [float(x) for x in SCORES], "e": [round(float(x), 2) for x in e], "esum": round(float(e.sum()), 2),
        "p": [round(float(x), 3) for x in p], "p_pct": [round(float(x) * 100) for x in p],
        "cold_pct": [round(float(x) * 100) for x in cold], "hot_pct": [round(float(x) * 100) for x in hot],
        "shift_check": [round(float(x), 3) for x in softmax(SCORES + 100)],
        "overflow_at": 710,
        "loss": round(L, 3), "grad": [round(float(x), 3) for x in g], "surprise_half": round(surprise(0.5), 3), "surprise_01": round(surprise(0.1), 2),
        "loss_after_4": round(rows[4][2], 3), "p_target_after_4": round(float(rows[4][1][TARGET]), 3),
        "entropy_uniform3": round(entropy([1 / 3] * 3), 3), "entropy_p": round(entropy(p), 3),
        "nt_the_1": round(nt1["the"] * 100), "nt_a_1": round(nt1["a"] * 100), "nt_zebra_1": round(nt1["zebra"] * 100, 3),
        "nt_the_03": round(nt3["the"] * 100), "nt_a_03": round(nt3["a"] * 100),
        "clf_acc_start": round(float(accs[0]) * 100), "clf_acc_end": round(float(accs[-1]) * 100), "clf_loss_start": round(float(losses[0]), 3), "clf_loss_end": round(float(losses[-1]), 3),
        "imagenet_classes": 1000, "imagenet_images_m": 1.2,
        **_model_facts(),
        **_derm_facts(),
    }


def _model_facts() -> dict:
    _, _, _, _, W, b, losses, accs = make_nodule_demo()
    return {
        "nod_acc_start": round(float(accs[0]) * 100), "nod_acc_end": round(float(accs[-1]) * 100),
        "nod_loss_start": round(float(losses[0]), 3), "nod_loss_end": round(float(losses[-1]), 3),
        "nod_small": (6.0, 0.3), "nod_large": (12.0, 0.6),
        "nod_risk_small_pct": round(nodule_risk(6.0, 0.3, W, b) * 100), "nod_risk_large_pct": round(nodule_risk(12.0, 0.6, W, b) * 100),
        "ardila_auc": ARDILA_2019["auc_pct"], "ardila_cases": ARDILA_2019["cases"], "ardila_fp": ARDILA_2019["fp_reduction_pct"],
        "ardila_fn": ARDILA_2019["fn_reduction_pct"], "ardila_radiologists": ARDILA_2019["radiologists"],
        "gpt2_vocab": GPT2_VOCAB, "ppl_example_ce": PPL_EXAMPLE_CE, "ppl_example": round(perplexity(PPL_EXAMPLE_CE), 1),
    }
