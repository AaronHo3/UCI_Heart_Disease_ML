"""Phase 1 — Leave-One-Site-Out (LOSO) external validation.

Research question
-----------------
Does a heart-disease classifier trained on some hospitals still work on a
*different* hospital it has never seen? This is the question that matters for
clinical deployment and the one a pooled random split silently ignores.

Method
------
1. **Pooled baseline (optimistic):** stratified K-fold CV over all four cohorts
   mixed together. Test folds contain rows from every site, so this is the
   in-distribution number most portfolio projects report.
2. **LOSO (realistic):** for each cohort, train on the other three and test on
   the held-out cohort. Per-site AUC + calibration are reported, plus a pooled
   estimate over all out-of-site predictions.
3. **Generalization gap:** pooled-CV AUC minus pooled-LOSO AUC. A positive gap
   is the finding: clinical ML degrades across sites.
4. **Cohort-leakage probe:** re-run the pooled baseline with the ``dataset``
   site indicator added as a feature. Because prevalence ranges 36%->93% across
   sites, a model that can see the site can shortcut to its base rate. Any AUC
   inflation demonstrates *why* the site column must stay out of the features.

Outputs
-------
* ``reports/loso_results.csv`` — every per-site / pooled metric with CIs.
* ``reports/figures/loso_gap.png`` — pooled vs LOSO AUC per model.
* ``reports/figures/loso_per_site.png`` — per-site held-out AUC with CIs.
* ``reports/figures/loso_calibration_drift.png`` — mean predicted prob vs true
  prevalence per held-out site (the label-shift miscalibration exhibit).
"""
from __future__ import annotations

import os
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from data import CohortData, cohort_profile, load_cohort_data
from metrics import Estimate, bootstrap_auc, summarize
from pipeline_utils import (
    RANDOM_SEED,
    build_hgb_pipeline,
    build_logreg_pipeline,
    build_rf_pipeline,
)

REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
N_SPLITS = 5
N_BOOT = 2000

# Model factories keyed by display name. Each takes the feature frame so the
# preprocessor can infer numeric/categorical columns.
MODEL_BUILDERS: dict[str, Callable[[pd.DataFrame], Pipeline]] = {
    "Logistic Regression": build_logreg_pipeline,
    "Random Forest": build_rf_pipeline,
    "Gradient Boosting": build_hgb_pipeline,
}


def pooled_cv_predictions(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> np.ndarray:
    """Out-of-fold positive-class probabilities from stratified K-fold CV."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=-1)
    return proba[:, 1]


def loso_predictions(
    builder: Callable[[pd.DataFrame], Pipeline], data: CohortData
) -> pd.DataFrame:
    """Hold out each site once; predict it from a model trained on the rest.

    Returns a long frame with columns: site, y_true, y_prob.
    """
    rows: list[pd.DataFrame] = []
    for held_out in data.sites:
        train_sites = [s for s in data.sites if s != held_out]
        train = data.for_sites(train_sites)
        test = data.for_sites([held_out])

        pipeline = clone(builder(train.X))
        pipeline.fit(train.X, train.y)
        prob = pipeline.predict_proba(test.X)[:, 1]
        rows.append(
            pd.DataFrame(
                {"site": held_out, "y_true": test.y.to_numpy(), "y_prob": prob}
            )
        )
    return pd.concat(rows, ignore_index=True)


def cohort_leakage_probe(
    builder: Callable[[pd.DataFrame], Pipeline], data: CohortData
) -> tuple[Estimate, Estimate]:
    """Pooled-CV AUC without vs with the site indicator as a feature."""
    without = bootstrap_auc(
        data.y.to_numpy(),
        pooled_cv_predictions(builder(data.X), data.X, data.y),
        n_boot=N_BOOT,
    )

    X_with_site = data.X.copy()
    X_with_site["dataset"] = data.site.to_numpy()  # re-inject as a feature
    with_site = bootstrap_auc(
        data.y.to_numpy(),
        pooled_cv_predictions(builder(X_with_site), X_with_site, data.y),
        n_boot=N_BOOT,
    )
    return without, with_site


def run_study(data: CohortData) -> dict[str, object]:
    """Execute the full Phase 1 analysis and return structured results."""
    results: list[dict] = []
    per_site_records: dict[str, pd.DataFrame] = {}
    gaps: list[dict] = []

    for model_name, builder in MODEL_BUILDERS.items():
        # 1. Pooled baseline (optimistic, in-distribution).
        pooled_prob = pooled_cv_predictions(builder(data.X), data.X, data.y)
        pooled = summarize(
            "POOLED-CV", data.y.to_numpy(), pooled_prob, n_boot=N_BOOT
        )
        row = pooled.as_row() | {"model": model_name, "regime": "pooled_cv"}
        results.append(row)

        # 2. LOSO (realistic, out-of-distribution).
        loso = loso_predictions(builder, data)
        per_site_records[model_name] = loso
        for site in data.sites:
            sub = loso[loso["site"] == site]
            rep = summarize(
                site, sub["y_true"].to_numpy(), sub["y_prob"].to_numpy(), n_boot=N_BOOT
            )
            results.append(
                rep.as_row() | {"model": model_name, "regime": "loso_site"}
            )

        loso_pooled = summarize(
            "LOSO-POOLED", loso["y_true"].to_numpy(), loso["y_prob"].to_numpy(),
            n_boot=N_BOOT,
        )
        results.append(
            loso_pooled.as_row() | {"model": model_name, "regime": "loso_pooled"}
        )

        # 3. Generalization gap.
        gap = pooled.auc.value - loso_pooled.auc.value
        gaps.append(
            {
                "model": model_name,
                "pooled_cv_auc": pooled.auc.value,
                "loso_pooled_auc": loso_pooled.auc.value,
                "generalization_gap": gap,
            }
        )

    # 4. Cohort-leakage probe (logistic regression: cleanest to interpret).
    without, with_site = cohort_leakage_probe(
        MODEL_BUILDERS["Logistic Regression"], data
    )

    return {
        "results": pd.DataFrame(results),
        "per_site": per_site_records,
        "gaps": pd.DataFrame(gaps),
        "leakage": {"without_site": without, "with_site": with_site},
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _ensure_dirs() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_gap(gaps: pd.DataFrame) -> str:
    _ensure_dirs()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(gaps))
    width = 0.38
    ax.bar(x - width / 2, gaps["pooled_cv_auc"], width, label="Pooled CV (optimistic)")
    ax.bar(x + width / 2, gaps["loso_pooled_auc"], width, label="LOSO (out-of-site)")
    for i, row in gaps.iterrows():
        ax.annotate(
            f"-{row['generalization_gap']:.3f}",
            xy=(i, row["loso_pooled_auc"]),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            color="darkred",
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(gaps["model"], rotation=10)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Generalization gap: pooled CV vs leave-one-site-out")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "loso_gap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_per_site(results: pd.DataFrame) -> str:
    _ensure_dirs()
    site_rows = results[results["regime"] == "loso_site"].copy()
    models = list(MODEL_BUILDERS)
    sites = sorted(site_rows["site"].unique())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.25
    x = np.arange(len(sites))
    for j, model in enumerate(models):
        sub = site_rows[site_rows["model"] == model].set_index("site").loc[sites]
        vals = sub["auc"].to_numpy()
        lo = vals - sub["auc_lo"].to_numpy()
        hi = sub["auc_hi"].to_numpy() - vals
        ax.bar(x + (j - 1) * width, vals, width, label=model, yerr=[lo, hi], capsize=3)

    # Annotate fragility: held-out positives count per site.
    counts = (
        site_rows[site_rows["model"] == models[0]].set_index("site").loc[sites]
    )
    labels = [
        f"{s}\n(n={int(counts.loc[s, 'n'])}, pos={int(counts.loc[s, 'n_pos'])})"
        for s in sites
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax.set_ylabel("Held-out ROC-AUC (95% bootstrap CI)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Per-site external validation (wide CIs = few held-out cases)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "loso_per_site.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_calibration_drift(per_site: dict[str, pd.DataFrame]) -> str:
    """Mean predicted probability vs true prevalence per held-out site.

    Under label shift, a model trained on low-prevalence sites systematically
    under-predicts on high-prevalence sites: points fall below the diagonal.
    """
    _ensure_dirs()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="perfect calibration")
    markers = {"Logistic Regression": "o", "Random Forest": "s", "Gradient Boosting": "^"}
    for model, loso in per_site.items():
        grp = loso.groupby("site")
        true_prev = grp["y_true"].mean()
        pred_mean = grp["y_prob"].mean()
        ax.scatter(true_prev, pred_mean, s=70, marker=markers[model], label=model)
        if model == "Gradient Boosting":
            for site in true_prev.index:
                ax.annotate(
                    site,
                    (true_prev[site], pred_mean[site]),
                    fontsize=8,
                    xytext=(5, -2),
                    textcoords="offset points",
                )
    ax.set_xlabel("True prevalence in held-out site")
    ax.set_ylabel("Mean predicted probability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Calibration drift under label shift (LOSO)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "loso_calibration_drift.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    data = load_cohort_data()
    print("=== Cohort profile (sorted by prevalence) ===")
    print(cohort_profile(data).round(3).to_string())
    print()

    study = run_study(data)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    results: pd.DataFrame = study["results"]
    out_csv = os.path.join(REPORTS_DIR, "loso_results.csv")
    results.to_csv(out_csv, index=False)

    print("=== Generalization gap (pooled CV vs LOSO) ===")
    print(study["gaps"].round(4).to_string(index=False))
    print()

    print("=== Per-site held-out AUC (Gradient Boosting) ===")
    gb = results[(results["model"] == "Gradient Boosting") & (results["regime"] == "loso_site")]
    print(gb[["site", "n", "n_pos", "prevalence", "auc", "auc_lo", "auc_hi", "ece"]].to_string(index=False))
    print()

    leak = study["leakage"]
    print("=== Cohort-leakage probe (Logistic Regression, pooled CV) ===")
    print(f"  without site feature: AUC {leak['without_site']}")
    print(f"  with    site feature: AUC {leak['with_site']}")
    inflation = leak["with_site"].value - leak["without_site"].value
    print(f"  inflation from leaking site: {inflation:+.4f}")
    print()

    paths = [
        plot_gap(study["gaps"]),
        plot_per_site(results),
        plot_calibration_drift(study["per_site"]),
    ]
    print("=== Wrote ===")
    print(f"  {out_csv}")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
