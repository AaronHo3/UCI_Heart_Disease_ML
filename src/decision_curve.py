"""Phase 3b - Decision Curve Analysis and a clinical baseline.

AUC and calibration still don't answer "is this model clinically *useful*?"
Decision Curve Analysis (Vickers & Elkin, 2006) does: it computes **net
benefit** across the range of threshold probabilities a clinician might act on,
and compares the model to the two trivial policies - treat everyone and treat
no one.

We also pit the full ML model against a deliberately simple **clinical risk
proxy**: a parsimonious logistic regression on routinely-available
history/exercise-test variables (age, sex, chest-pain type, resting BP, max
heart rate, exercise angina, ST depression). This is NOT a validated score
(Framingham/ASCVD need HDL, smoking, BP-treatment and diabetes detail this
dataset lacks, and Switzerland has no usable cholesterol). It is an honest
"how much does the complex model actually add over something simple?" baseline.

Outputs
-------
* ``reports/decision_curve.csv`` - net benefit by threshold.
* ``reports/clinical_baseline.csv`` - AUC of full model vs proxy + DeLong test.
* ``reports/figures/decision_curve.png`` - net benefit curves.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

from data import load_cohort_data
from pipeline_utils import build_hgb_pipeline, build_logreg_pipeline, split_train_test
from stats import delong_roc_test
from metrics import bootstrap_auc

REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Routinely-available, clinically standard, reasonably complete across cohorts.
# Excludes ca/thal/slope (effectively Cleveland-only) so the proxy is deployable.
PROXY_FEATURES = ["age", "sex", "cp", "trestbps", "thalch", "exang", "oldpeak"]


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Net benefit = TP/n - FP/n * (pt / (1 - pt)) across threshold probs."""
    y_true = np.asarray(y_true)
    n = len(y_true)
    nb = np.empty_like(thresholds, dtype=float)
    for i, pt in enumerate(thresholds):
        pred = y_prob >= pt
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        nb[i] = tp / n - (fp / n) * (pt / (1 - pt))
    return nb


def treat_all_net_benefit(prevalence: float, thresholds: np.ndarray) -> np.ndarray:
    return prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))


def run(data) -> dict:
    X_tr, X_te, y_tr, y_te = split_train_test(data.X, data.y)
    y_te_arr = y_te.to_numpy()

    # Full model: Gradient Boosting on all features.
    full = clone(build_hgb_pipeline(X_tr)).fit(X_tr, y_tr)
    full_prob = full.predict_proba(X_te)[:, 1]

    # Clinical proxy: parsimonious logistic regression on routine features.
    Xp_tr, Xp_te = X_tr[PROXY_FEATURES], X_te[PROXY_FEATURES]
    proxy = clone(build_logreg_pipeline(Xp_tr)).fit(Xp_tr, y_tr)
    proxy_prob = proxy.predict_proba(Xp_te)[:, 1]

    thresholds = np.linspace(0.01, 0.95, 95)
    prevalence = float(y_te_arr.mean())
    curves = pd.DataFrame(
        {
            "threshold": thresholds,
            "full_model": net_benefit(y_te_arr, full_prob, thresholds),
            "clinical_proxy": net_benefit(y_te_arr, proxy_prob, thresholds),
            "treat_all": treat_all_net_benefit(prevalence, thresholds),
            "treat_none": 0.0,
        }
    )

    full_auc = bootstrap_auc(y_te_arr, full_prob)
    proxy_auc = bootstrap_auc(y_te_arr, proxy_prob)
    delong = delong_roc_test(y_te_arr, full_prob, proxy_prob)
    baseline = pd.DataFrame(
        [
            {
                "model": "Full ML (GB, all features)",
                "n_features": data.X.shape[1],
                "auc": round(full_auc.value, 4),
                "auc_lo": round(full_auc.lower, 4),
                "auc_hi": round(full_auc.upper, 4),
            },
            {
                "model": f"Clinical proxy (LogReg, {len(PROXY_FEATURES)} routine features)",
                "n_features": len(PROXY_FEATURES),
                "auc": round(proxy_auc.value, 4),
                "auc_lo": round(proxy_auc.lower, 4),
                "auc_hi": round(proxy_auc.upper, 4),
            },
        ]
    )
    return {"curves": curves, "baseline": baseline, "delong": delong, "prevalence": prevalence}


def plot_decision_curve(curves: pd.DataFrame) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(curves["threshold"], curves["full_model"], label="Full ML model (GB)", lw=2)
    ax.plot(curves["threshold"], curves["clinical_proxy"], label="Clinical proxy (LogReg)", lw=2)
    ax.plot(curves["threshold"], curves["treat_all"], "--", color="grey", label="Treat all")
    ax.axhline(0.0, color="black", lw=1, label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    # Net benefit below 0 is clinically meaningless; clip the view.
    ax.set_ylim(-0.05, curves[["full_model", "clinical_proxy", "treat_all"]].to_numpy().max() * 1.1)
    ax.set_title("Decision Curve Analysis (held-out test split)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "decision_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    data = load_cohort_data()
    res = run(data)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    res["curves"].to_csv(os.path.join(REPORTS_DIR, "decision_curve.csv"), index=False)
    res["baseline"].to_csv(os.path.join(REPORTS_DIR, "clinical_baseline.csv"), index=False)
    fig = plot_decision_curve(res["curves"])

    print(f"Test-split prevalence: {res['prevalence']:.3f}")
    print("\n=== Full ML model vs clinical proxy (AUC, 95% CI) ===")
    print(res["baseline"].to_string(index=False))
    d = res["delong"]
    print(
        f"\nDeLong full vs proxy: delta={d.diff:+.4f}, p={d.p_value:.4f} -> "
        f"{d.verdict()}"
    )
    print(f"\nWrote: {fig}, reports/decision_curve.csv, reports/clinical_baseline.csv")


if __name__ == "__main__":
    main()
