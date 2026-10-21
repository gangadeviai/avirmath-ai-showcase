# Verified results

Every number below is produced by the modules in `src/avirmath/mathcore/`, is covered by a test in
`tests/`, and is the same number the videos speak and typeset. Re-create all of it with
`python -m pytest -q` (125 tests, about two minutes).

## Test suite

| Module | Tests cover |
|---|---|
| `vectors` | dot product, projection, cosine, 3-D geometry used in the vectors lesson |
| `matrices` | matrix-vector maps, composition, determinant, eigenvectors, PageRank, low-rank/LoRA storage counts, GPT-2 layer sizes |
| `gradient_descent` | analytic vs finite-difference gradients, step-size behaviour, convergence on the lesson landscape |
| `backprop` | chain rule through a two-layer network against SymPy, gradient checks |
| `softmax` | softmax/cross-entropy identities, gradient `p - y`, perplexity, the nodule demo, the dermatology model and its failure modes |
| `regression` | closed-form and gradient fits, AUC and ECE definitions, the credit model, the bike model and its failure modes |
| `adaptive_filter` | LMS adaptive filter used in the gradient-descent lesson |

Result on 21 October 2026: **125 passed**.

## Real datasets modelled on screen

### German Credit (UCI), logistic regression, lesson *linear-logistic-regression*

| Quantity | Value |
|---|---|
| Rows / features used | 1000 / 8 |
| Train / test split | 800 / 200 |
| Accuracy vs always-"good" baseline | 74 % vs 71 % |
| AUC | 0.74 |
| Expected calibration error | 0.074 |
| Recall of bad loans at threshold 0.5 / 0.3 | 48 % / 65 % |
| Applicants flagged at 0.5 / 0.3 (of 200) | 49 / 87 |
| Odds ratio: checking-account status | 0.47 |
| Odds ratio: loan duration | 1.35 |
| Odds ratio: loan amount | 1.27 |

### Bike Sharing hourly demand (UCI), linear regression, lesson *linear-logistic-regression*

| Quantity | Value |
|---|---|
| Rows | 17379 (train 13903, test 3476) |
| RMSE / MAE / R² on held-out hours | 140 / 97 / 0.42 |
| Failure shown: target leakage gives RMSE | 0.0 |
| Failure shown: extrapolating to 60 °C predicts | 577 rides (vs 375 at 30 °C) |
| Failure shown: residual at 08:00 / 17:00 / 03:00 | +237 / +160 / -34 rides |

### Dermatology (UCI), six-class softmax classifier, lesson *softmax-cross-entropy*

| Quantity | Value |
|---|---|
| Patients / features / classes | 366 / 34 / 6 |
| Accuracy / top-2 accuracy on held-out patients | 97 % / 100 % |
| Recall on the rarest class (pityriasis rubra pilaris, n = 20) | 100 % |
| Expected calibration error | 0.018 |
| Published reference accuracy (Güvenir et al., 1998) | 96.2 % |
| Failure shown: over-trained, no regulariser | confidence 98.7 %, accuracy 96 %, ECE 0.041 |
| Failure shown: clinical features only (no biopsy) | accuracy 92 % |
| Failure shown: 20 % label noise | accuracy 86 % |

## Production quality gates (per lesson)

| Gate | Rule | Result on all six lessons |
|---|---|---|
| Facts | every spoken or typeset number resolves to `lesson_facts()` | 100 % |
| Equations | every equation names the test that checks it | all resolve |
| Sync | no animation beat runs longer than its narration | overrun 0 s |
| Frames | one QA still per beat, inspected | 0 defects |
| Voice | spectral energy 5–8 kHz after clean-up | 1.78 % → 0.03 % |

## Published lessons

| Lesson | Length | Chapters |
|---|---|---|
| Gradient descent | 5.7 min | 8 |
| Backpropagation | 5.2 min | 8 |
| Vectors and the dot product | 7.4 min | 9 |
| Matrices in AI | 7.1 min | 8 |
| Softmax and cross-entropy | 9.6 min | 9 |
| Linear vs logistic regression | 9.8 min | 8 |
