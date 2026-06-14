"""Reusable evaluation metrics with uncertainty quantification.

These helpers are deliberately dependency-light (numpy + scikit-learn only) so
they can back the LOSO study (Phase 1), the statistical-rigor work (Phase 2),
and the calibration analysis (Phase 3) without pulling in heavyweight packages.

Design notes
------------
* Discrimination metrics (AUC) are reported with bootstrap confidence
  intervals, never as bare point estimates.
* Calibration is summarised with the Brier score and Expected Calibration
  Error (ECE), which together capture "are the predicted probabilities
  trustworthy", a question AUC cannot answer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

RANDOM_SEED = 100


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a (1 - alpha) bootstrap confidence interval."""

    value: float
    lower: float
    upper: float
    n: int

    def __str__(self) -> str:  # pragma: no cover - formatting helper
        return f"{self.value:.3f} [{self.lower:.3f}, {self.upper:.3f}]"


def _can_score(y_true: np.ndarray) -> bool:
    """AUC is undefined unless both classes are present."""
    return len(np.unique(y_true)) >= 2


def bootstrap_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_seed: int = RANDOM_SEED,
) -> Estimate:
    """Stratified bootstrap percentile CI for ROC-AUC.

    Resamples with replacement. Bootstrap replicates that happen to contain a
    single class are skipped (AUC undefined), which is why very imbalanced
    held-out sites (e.g. Switzerland, 8 negatives) yield wide, honest CIs.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if not _can_score(y_true):
        return Estimate(float("nan"), float("nan"), float("nan"), len(y_true))

    point = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(random_seed)
    n = len(y_true)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if not _can_score(yt):
            continue
        boots.append(roc_auc_score(yt, y_score[idx]))

    if not boots:
        return Estimate(point, float("nan"), float("nan"), n)

    lower = float(np.percentile(boots, 100 * alpha / 2))
    upper = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return Estimate(point, lower, upper, n)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error using equal-width probability bins.

    ECE = sum_b (n_b / N) * |accuracy_b - confidence_b|. Lower is better; 0
    means predicted probabilities match observed frequencies.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize: bin index in [1, n_bins]; clip the rightmost edge into bin n.
    bin_ids = np.clip(np.digitize(y_prob, bins[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        accuracy = float(y_true[mask].mean())
        confidence = float(y_prob[mask].mean())
        ece += (count / n) * abs(accuracy - confidence)
    return float(ece)


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (mean squared error of probabilistic predictions)."""
    return float(brier_score_loss(np.asarray(y_true), np.asarray(y_prob)))


@dataclass(frozen=True)
class SiteReport:
    """Discrimination + calibration summary for one evaluation set."""

    name: str
    n: int
    n_pos: int
    prevalence: float
    auc: Estimate
    brier: float
    ece: float

    def as_row(self) -> dict[str, float | str]:
        return {
            "site": self.name,
            "n": self.n,
            "n_pos": self.n_pos,
            "prevalence": round(self.prevalence, 4),
            "auc": round(self.auc.value, 4),
            "auc_lo": round(self.auc.lower, 4),
            "auc_hi": round(self.auc.upper, 4),
            "brier": round(self.brier, 4),
            "ece": round(self.ece, 4),
        }


def summarize(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 2000,
    random_seed: int = RANDOM_SEED,
) -> SiteReport:
    """Bundle discrimination + calibration for an evaluation set."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_pos = int(y_true.sum())
    return SiteReport(
        name=name,
        n=len(y_true),
        n_pos=n_pos,
        prevalence=float(y_true.mean()) if len(y_true) else float("nan"),
        auc=bootstrap_auc(y_true, y_prob, n_boot=n_boot, random_seed=random_seed),
        brier=brier(y_true, y_prob) if _can_score(y_true) else float("nan"),
        ece=expected_calibration_error(y_true, y_prob),
    )
