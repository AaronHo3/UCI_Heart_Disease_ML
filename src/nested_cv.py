"""Phase 2 - Nested cross-validation, bootstrap CIs, and DeLong tests.

Three things every defensible model-comparison needs, which the original
single-CV pipeline lacked:

1. **Nested CV** so hyperparameter tuning cannot optimistically bias the
   reported score. The optimism is quantified by also reporting the naive
   "tune-and-report-on-the-same-folds" estimate.
2. **Bootstrap 95% CIs** on every headline metric (AUC, accuracy, F1,
   sensitivity, specificity) - no naked point estimates.
3. **DeLong's test** for whether AUC differences between models are real. On
   this dataset they are expected to be statistically indistinguishable, and
   reporting that honestly is the point.

Outputs
-------
* ``reports/nested_cv_results.csv`` - metric ± CI per model.
* ``reports/delong_pairwise.csv`` - pairwise AUC tests.
* ``reports/figures/nested_vs_naive_auc.png`` - the optimism-bias exhibit.

The narrative summary lives in the single consolidated ``reports/FINDINGS.md``.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from data import load_cohort_data
from pipeline_utils import (
    RANDOM_SEED,
    build_hgb_pipeline,
    build_logreg_pipeline,
    build_rf_pipeline,
)
from stats import bootstrap_classification_metrics, delong_roc_test

REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
OUTER_SPLITS = 5
INNER_SPLITS = 3
N_ITER = 8
N_BOOT = 2000
THRESHOLD = 0.5  # report thresholded metrics at the default operating point

# (builder, param distribution). Grids are deliberately small so nested CV
# stays tractable while still exercising real tuning.
MODEL_SPECS = {
    "Logistic Regression": (
        build_logreg_pipeline,
        {
            "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    "Random Forest": (
        build_rf_pipeline,
        {
            "clf__n_estimators": [200, 300],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_leaf": [1, 3, 5],
            "clf__max_features": ["sqrt", 0.5],
        },
    ),
    "Gradient Boosting": (
        build_hgb_pipeline,
        {
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__max_leaf_nodes": [15, 31],
            "clf__max_depth": [None, 3],
            "clf__l2_regularization": [0.0, 1.0],
        },
    ),
}


def _make_search(builder, X, grid) -> RandomizedSearchCV:
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    return RandomizedSearchCV(
        estimator=builder(X),
        param_distributions=grid,
        n_iter=N_ITER,
        scoring="roc_auc",
        cv=inner,
        random_state=RANDOM_SEED,
        n_jobs=1,
        refit=True,
    )


def run_nested_cv(X, y) -> dict[str, object]:
    outer = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_proba: dict[str, np.ndarray] = {}
    naive_auc: dict[str, float] = {}

    for name, (builder, grid) in MODEL_SPECS.items():
        search = _make_search(builder, X, grid)

        # Nested (unbiased): tuning happens inside each outer fold.
        proba = cross_val_predict(
            search, X, y, cv=outer, method="predict_proba", n_jobs=-1
        )[:, 1]
        oof_proba[name] = proba

        # Naive (biased): tune on all data, report the best inner-CV score.
        search_full = _make_search(builder, X, grid)
        search_full.fit(X, y)
        naive_auc[name] = float(search_full.best_score_)

    return {"oof_proba": oof_proba, "naive_auc": naive_auc}


def build_metric_table(y, oof_proba: dict[str, np.ndarray], naive_auc: dict[str, float]) -> pd.DataFrame:
    rows = []
    y_arr = np.asarray(y)
    for name, proba in oof_proba.items():
        panel = bootstrap_classification_metrics(y_arr, proba, threshold=THRESHOLD, n_boot=N_BOOT)
        row = {"model": name, "naive_auc": round(naive_auc[name], 4)}
        for metric, est in panel.items():
            row[metric] = round(est.value, 4)
            row[f"{metric}_lo"] = round(est.lower, 4)
            row[f"{metric}_hi"] = round(est.upper, 4)
        row["optimism_bias"] = round(naive_auc[name] - panel["auc"].value, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_delong(y, oof_proba: dict[str, np.ndarray]) -> pd.DataFrame:
    y_arr = np.asarray(y)
    names = list(oof_proba)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            res = delong_roc_test(y_arr, oof_proba[a], oof_proba[b])
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "auc_a": round(res.auc_a, 4),
                    "auc_b": round(res.auc_b, 4),
                    "auc_diff": round(res.diff, 4),
                    "z": round(res.z, 3),
                    "p_value": round(res.p_value, 4),
                    "verdict": res.verdict(),
                }
            )
    return pd.DataFrame(rows)


def plot_optimism(table: pd.DataFrame) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(table))
    width = 0.38
    ax.bar(x - width / 2, table["naive_auc"], width, label="Naive CV (tuning peeks)")
    ax.bar(
        x + width / 2,
        table["auc"],
        width,
        yerr=[table["auc"] - table["auc_lo"], table["auc_hi"] - table["auc"]],
        capsize=4,
        label="Nested CV (unbiased) ±95% CI",
    )
    for i, row in table.iterrows():
        ax.annotate(
            f"+{row['optimism_bias']:.3f}",
            xy=(i - width / 2, row["naive_auc"]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="darkred",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(table["model"], rotation=10)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Optimism bias: naive vs nested cross-validation")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "nested_vs_naive_auc.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    data = load_cohort_data()
    print(f"Pooled n={len(data.y)}, prevalence={data.y.mean():.3f}")
    res = run_nested_cv(data.X, data.y)

    table = build_metric_table(data.y, res["oof_proba"], res["naive_auc"])
    delong = pairwise_delong(data.y, res["oof_proba"])

    os.makedirs(REPORTS_DIR, exist_ok=True)
    table.to_csv(os.path.join(REPORTS_DIR, "nested_cv_results.csv"), index=False)
    delong.to_csv(os.path.join(REPORTS_DIR, "delong_pairwise.csv"), index=False)
    fig = plot_optimism(table)

    print("\n=== Nested CV metrics (point | 95% CI) ===")
    show = ["model", "auc", "auc_lo", "auc_hi", "naive_auc", "optimism_bias"]
    print(table[show].to_string(index=False))
    print("\n=== DeLong pairwise ===")
    print(delong.to_string(index=False))
    print(f"\nWrote: {fig}, reports/nested_cv_results.csv, reports/delong_pairwise.csv")


if __name__ == "__main__":
    main()
