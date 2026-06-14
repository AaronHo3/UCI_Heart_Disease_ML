"""Statistical inference helpers: DeLong's test and bootstrap metric CIs.

Why these exist
---------------
"Model A beats Model B" is meaningless without a test. On a famous dataset,
two models with AUCs of 0.88 and 0.86 are almost always *statistically
indistinguishable*, and saying so plainly is a marker of scientific maturity.

* :func:`delong_roc_test` compares two ROC-AUCs computed on the **same** samples
  (correlated), using the fast DeLong algorithm (Sun & Xu, 2014). It returns the
  two-sided p-value for H0: AUC_1 == AUC_2.
* :func:`bootstrap_classification_metrics` returns point estimates with 95%
  bootstrap percentile CIs for AUC, accuracy, F1, sensitivity, and specificity.

The DeLong implementation is a clean reimplementation of the standard public
algorithm (no external dependency beyond numpy/scipy).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from metrics import Estimate

RANDOM_SEED = 100


# --------------------------------------------------------------------------- #
# DeLong's test for two correlated ROC AUCs
# --------------------------------------------------------------------------- #
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Mid-ranks with ties averaged (1-based)."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks + 1
    return out


def _fast_delong(preds: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong covariance for k predictors.

    ``preds`` is k x N with the positive samples in the first ``n_pos`` columns.
    Returns (aucs[k], covariance[k, k]).
    """
    m = n_pos
    n = preds.shape[1] - m
    pos = preds[:, :m]
    neg = preds[:, m:]
    k = preds.shape[0]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


@dataclass(frozen=True)
class DelongResult:
    auc_a: float
    auc_b: float
    diff: float
    z: float
    p_value: float

    def verdict(self, alpha: float = 0.05) -> str:
        return (
            "significant" if self.p_value < alpha else "indistinguishable"
        )


def delong_roc_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> DelongResult:
    """Two-sided DeLong test for AUC(score_a) vs AUC(score_b) on shared samples."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        raise ValueError("DeLong test requires both classes present.")

    order = (-y_true).argsort(kind="mergesort")  # positives (label 1) first, stable
    n_pos = int(y_true.sum())
    preds = np.vstack((np.asarray(score_a), np.asarray(score_b)))[:, order]

    aucs, cov = _fast_delong(preds, n_pos)
    contrast = np.array([[1.0, -1.0]])
    var = float((contrast @ cov @ contrast.T).item())
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        # Identical (or degenerate) predictions -> no detectable difference.
        z, p = 0.0, 1.0
    else:
        z = diff / np.sqrt(var)
        p = float(2 * stats.norm.sf(abs(z)))
    return DelongResult(float(aucs[0]), float(aucs[1]), diff, float(z), p)


# --------------------------------------------------------------------------- #
# Bootstrap CIs for a full panel of classification metrics
# --------------------------------------------------------------------------- #
def _sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return float(sens), float(spec)


def _metric_panel(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    sens, spec = _sensitivity_specificity(y_true, y_pred)
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": sens,
        "specificity": spec,
    }


def bootstrap_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Estimate]:
    """Point estimate + percentile bootstrap CI for a panel of metrics."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    point = _metric_panel(y_true, y_prob, threshold)

    rng = np.random.default_rng(random_seed)
    n = len(y_true)
    samples: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        panel = _metric_panel(yt, y_prob[idx], threshold)
        for k, v in panel.items():
            samples[k].append(v)

    out: dict[str, Estimate] = {}
    for k, v in point.items():
        arr = np.asarray(samples[k], dtype=float)
        if arr.size == 0:
            out[k] = Estimate(v, float("nan"), float("nan"), n)
        else:
            lo = float(np.percentile(arr, 100 * alpha / 2))
            hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
            out[k] = Estimate(float(v), lo, hi, n)
    return out
