"""Phase 4 - Interpretability with clinical face validity.

A clinical model is only trustworthy if its learned effects agree with
established cardiology (and the disagreements are interpretable). This module:

* explains the Gradient Boosting model with exact tree SHAP (global beeswarm +
  two local patient explanations),
* corroborates with model-agnostic permutation importance (with error bars) and
  partial-dependence plots,
* cross-checks the *direction* of each effect against known risk factors using
  the logistic-regression coefficients (signed, easy to read).

Outputs
-------
* ``reports/figures/shap_summary.png``, ``shap_local.png``
* ``reports/figures/permutation_importance.png``, ``pdp.png``
* ``reports/permutation_importance.csv``, ``reports/clinical_crosscheck.csv``
"""
from __future__ import annotations

import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from data import load_cohort_data
from pipeline_utils import build_hgb_pipeline, build_logreg_pipeline, split_train_test

warnings.filterwarnings("ignore")

REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
N_REPEATS = 30
RANDOM_SEED = 100

# Direction each feature should push risk, per established cardiology.
# (+1 = higher value -> more disease; -1 = higher value -> less disease.)
CLINICAL_PRIOR = {
    "oldpeak": (+1, "ST depression on exercise -> ischemia"),
    "ca": (+1, "more major vessels affected -> more disease"),
    "exang_True": (+1, "exercise-induced angina -> more disease"),
    "thalch": (-1, "higher max heart rate -> better exercise capacity"),
    "age": (+1, "older age -> more disease"),
    "sex_Male": (+1, "male sex -> higher CAD risk in this cohort"),
    "cp_asymptomatic": (+1, "asymptomatic presentation flags silent disease here"),
    "thal_reversable defect": (+1, "reversible perfusion defect -> ischemia"),
}


def _clean(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "")


def _split_pipeline(builder, X_tr, y_tr):
    pipe = clone(builder(X_tr)).fit(X_tr, y_tr)
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    feats = [_clean(f) for f in prep.get_feature_names_out()]
    return pipe, prep, clf, feats


# --------------------------------------------------------------------------- #
# SHAP (Gradient Boosting)
# --------------------------------------------------------------------------- #
def shap_explanations(X_tr, y_tr, X_te) -> tuple[str, str]:
    _, prep, clf, feats = _split_pipeline(build_hgb_pipeline, X_tr, y_tr)
    X_te_t = prep.transform(X_te)

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_te_t)
    base = explainer.expected_value
    base = float(np.ravel(base)[0])
    expl = shap.Explanation(
        values=sv, base_values=np.full(len(sv), base), data=X_te_t, feature_names=feats
    )

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Global beeswarm.
    plt.figure()
    shap.summary_plot(sv, X_te_t, feature_names=feats, show=False, max_display=12)
    plt.title("SHAP global feature effects (Gradient Boosting)")
    plt.tight_layout()
    summary_path = os.path.join(FIGURES_DIR, "shap_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Two local explanations: highest- and lowest-risk test patients.
    risk = sv.sum(axis=1)
    hi, lo = int(np.argmax(risk)), int(np.argmin(risk))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, idx, label in ((axes[0], hi, "highest-risk"), (axes[1], lo, "lowest-risk")):
        plt.sca(ax)
        shap.plots.waterfall(expl[idx], max_display=8, show=False)
        ax.set_title(f"Local explanation: {label} patient")
    plt.tight_layout()
    local_path = os.path.join(FIGURES_DIR, "shap_local.png")
    plt.savefig(local_path, dpi=150, bbox_inches="tight")
    plt.close()
    return summary_path, local_path


# --------------------------------------------------------------------------- #
# Permutation importance (model-agnostic, with error bars)
# --------------------------------------------------------------------------- #
def permutation_importances(X_tr, y_tr, X_te, y_te) -> tuple[pd.DataFrame, str]:
    pipe = clone(build_hgb_pipeline(X_tr)).fit(X_tr, y_tr)
    result = permutation_importance(
        pipe, X_te, y_te, scoring="roc_auc", n_repeats=N_REPEATS, random_state=RANDOM_SEED, n_jobs=-1
    )
    table = (
        pd.DataFrame(
            {
                "feature": X_te.columns,
                "importance": result.importances_mean.round(4),
                "std": result.importances_std.round(4),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    top = table.iloc[::-1]
    ax.barh(top["feature"], top["importance"], xerr=top["std"], capsize=3)
    ax.set_xlabel("Mean AUC drop when permuted (+/-std over 30 repeats)")
    ax.set_title("Permutation importance (Gradient Boosting, held-out test)")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "permutation_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return table, path


# --------------------------------------------------------------------------- #
# Partial dependence for top continuous features
# --------------------------------------------------------------------------- #
def partial_dependence_plot(X_tr, y_tr) -> str:
    pipe = clone(build_hgb_pipeline(X_tr)).fit(X_tr, y_tr)
    # Well-populated continuous features with genuine marginal structure.
    # Heavily-missing features (thalch, chol, ca) have ~flat marginal PD because
    # most rows are imputed, so SHAP (which attributes per-row) is the right lens
    # for them - see shap_summary.png.
    features = ["oldpeak", "age", "trestbps"]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    PartialDependenceDisplay.from_estimator(pipe, X_tr, features, ax=ax, n_cols=3)
    fig.suptitle("Partial dependence (Gradient Boosting)", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "pdp.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Clinical cross-check (signed effects from logistic regression)
# --------------------------------------------------------------------------- #
def clinical_crosscheck(X_tr, y_tr) -> pd.DataFrame:
    _, _, clf, feats = _split_pipeline(build_logreg_pipeline, X_tr, y_tr)
    coefs = dict(zip(feats, clf.coef_[0]))
    rows = []
    for feat, (expected_sign, rationale) in CLINICAL_PRIOR.items():
        if feat not in coefs:
            continue
        coef = coefs[feat]
        learned_sign = int(np.sign(coef))
        rows.append(
            {
                "feature": feat,
                "logreg_coef": round(coef, 3),
                "learned_direction": "+" if learned_sign > 0 else "-",
                "expected_direction": "+" if expected_sign > 0 else "-",
                "agrees": learned_sign == expected_sign,
                "clinical_rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    data = load_cohort_data()
    X_tr, X_te, y_tr, y_te = split_train_test(data.X, data.y)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    summary_path, local_path = shap_explanations(X_tr, y_tr, X_te)
    perm_table, perm_path = permutation_importances(X_tr, y_tr, X_te, y_te)
    pdp_path = partial_dependence_plot(X_tr, y_tr)
    cross = clinical_crosscheck(X_tr, y_tr)

    perm_table.to_csv(os.path.join(REPORTS_DIR, "permutation_importance.csv"), index=False)
    cross.to_csv(os.path.join(REPORTS_DIR, "clinical_crosscheck.csv"), index=False)

    print("=== Permutation importance (top 8) ===")
    print(perm_table.head(8).to_string(index=False))
    print("\n=== Clinical cross-check (logistic regression signs) ===")
    print(cross.to_string(index=False))
    print(f"\n  Agreement: {int(cross['agrees'].sum())}/{len(cross)} features match cardiology priors")
    print(f"\nWrote: {summary_path}, {local_path}, {perm_path}, {pdp_path}")
    print("       reports/permutation_importance.csv, reports/clinical_crosscheck.csv")


if __name__ == "__main__":
    main()
