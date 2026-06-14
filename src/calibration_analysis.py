"""Phase 3a — Probability calibration.

AUC says nothing about whether a predicted "0.8" means an 80% chance of disease.
For a clinical tool that question is central: a miscalibrated model gives
misleading risk estimates even when it ranks patients perfectly.

This module evaluates calibration on a held-out test split (the test set is
never used for fitting or recalibration) and reports Brier score + Expected
Calibration Error (ECE) for each model:

* uncalibrated,
* after Platt scaling (sigmoid),
* after isotonic regression.

Outputs
-------
* ``reports/calibration_metrics.csv`` — Brier/ECE per model per method.
* ``reports/figures/calibration_reliability.png`` — reliability diagrams.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from data import load_cohort_data
from metrics import brier, expected_calibration_error
from pipeline_utils import (
    build_hgb_pipeline,
    build_logreg_pipeline,
    build_rf_pipeline,
    split_train_test,
)

REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
N_BINS = 10

MODEL_BUILDERS = {
    "Logistic Regression": build_logreg_pipeline,
    "Random Forest": build_rf_pipeline,
    "Gradient Boosting": build_hgb_pipeline,
}


def _probs_for_methods(builder, X_tr, y_tr, X_te) -> dict[str, np.ndarray]:
    """Return test-set probabilities: uncalibrated, sigmoid, isotonic."""
    base = clone(builder(X_tr)).fit(X_tr, y_tr)
    out = {"uncalibrated": base.predict_proba(X_te)[:, 1]}
    for method in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(clone(builder(X_tr)), method=method, cv=5)
        cal.fit(X_tr, y_tr)
        out[method] = cal.predict_proba(X_te)[:, 1]
    return out


def run_calibration(data) -> tuple[pd.DataFrame, dict]:
    X_tr, X_te, y_tr, y_te = split_train_test(data.X, data.y)
    rows = []
    probs_by_model: dict[str, dict[str, np.ndarray]] = {}
    for name, builder in MODEL_BUILDERS.items():
        probs = _probs_for_methods(builder, X_tr, y_tr, X_te)
        probs_by_model[name] = probs
        for method, p in probs.items():
            rows.append(
                {
                    "model": name,
                    "method": method,
                    "brier": round(brier(y_te.to_numpy(), p), 4),
                    "ece": round(expected_calibration_error(y_te.to_numpy(), p, N_BINS), 4),
                }
            )
    return pd.DataFrame(rows), {"y_te": y_te.to_numpy(), "probs": probs_by_model}


def plot_reliability(payload: dict) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    y_te = payload["y_te"]
    probs = payload["probs"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: all three models, uncalibrated.
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, pdict in probs.items():
        frac_pos, mean_pred = calibration_curve(y_te, pdict["uncalibrated"], n_bins=N_BINS, strategy="quantile")
        ax1.plot(mean_pred, frac_pos, marker="o", label=name)
    ax1.set_title("Reliability (uncalibrated)")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Observed frequency")
    ax1.legend(fontsize=8)

    # Right: best-discriminating model before/after recalibration.
    focus = "Gradient Boosting"
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for method, style in (("uncalibrated", "o-"), ("sigmoid", "s-"), ("isotonic", "^-")):
        frac_pos, mean_pred = calibration_curve(y_te, probs[focus][method], n_bins=N_BINS, strategy="quantile")
        ax2.plot(mean_pred, frac_pos, style, label=method)
    ax2.set_title(f"Recalibration effect ({focus})")
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Observed frequency")
    ax2.legend(fontsize=8)

    fig.suptitle("Calibration on held-out test split", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "calibration_reliability.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    data = load_cohort_data()
    table, payload = run_calibration(data)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    table.to_csv(os.path.join(REPORTS_DIR, "calibration_metrics.csv"), index=False)
    fig = plot_reliability(payload)

    print("=== Calibration metrics (held-out test split) ===")
    print(table.to_string(index=False))
    print(f"\nWrote: {fig}, reports/calibration_metrics.csv")


if __name__ == "__main__":
    main()
