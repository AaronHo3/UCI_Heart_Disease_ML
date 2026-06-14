"""Phase 6 — Uncertainty quantification via split-conformal prediction.

Standard ML gives a point probability with no guarantee. Split-conformal
prediction wraps any fitted model and returns, for each patient, a *prediction
set* drawn from {no-disease, disease} with a finite-sample coverage guarantee:
the true label is in the set with probability >= 1 - alpha, regardless of the
model being right.

Method (Least Ambiguous set-valued Classifier, LAC)
---------------------------------------------------
1. Split: train / calibration / test (disjoint).
2. Fit the model on train.
3. Nonconformity score on calibration = 1 - p(true class).
4. q_hat = the conformal quantile of calibration scores at level
   ceil((n_cal+1)(1-alpha)) / n_cal.
5. Test prediction set = { class c : p(c) >= 1 - q_hat }.

Sets of size 2 ({both}) are the model abstaining — clinically, "refer for more
testing". We then verify empirical coverage and expose a known caveat: split
conformal guarantees *marginal* coverage but can under-cover a subgroup, which
links back to the Phase 7 fairness finding.

Outputs
-------
* ``reports/conformal_coverage.csv``
* ``reports/figures/conformal_coverage.png``
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from data import load_cohort_data
from pipeline_utils import RANDOM_SEED, build_logreg_pipeline

REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
ALPHAS = [0.1, 0.2]
# Coverage is guaranteed in expectation over the calibration/test split, so we
# verify it by averaging over many random splits rather than trusting one.
N_REPEATS = 100


def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    n = len(cal_scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(cal_scores, level, method="higher"))


def prediction_sets(proba: np.ndarray, q_hat: float) -> np.ndarray:
    """Boolean (n, 2) inclusion matrix: include class c if p(c) >= 1 - q_hat."""
    return proba >= (1 - q_hat)


def _set_size_breakdown(sets: np.ndarray) -> dict[str, float]:
    sizes = sets.sum(axis=1)
    return {
        "mean_set_size": round(float(sizes.mean()), 3),
        "pct_singleton": round(float((sizes == 1).mean()), 3),
        "pct_uncertain": round(float((sizes == 2).mean()), 3),  # {0,1}
        "pct_empty": round(float((sizes == 0).mean()), 3),
    }


def _coverage(sets: np.ndarray, y_true: np.ndarray) -> float:
    covered = sets[np.arange(len(y_true)), y_true.astype(int)]
    return float(covered.mean())


def _one_split(data, seed: int) -> list[dict]:
    """Fit on one random train/cal/test split; return per-alpha records."""
    X_fit, X_te, y_fit, y_te = train_test_split(
        data.X, data.y, test_size=0.25, random_state=seed, stratify=data.y
    )
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_fit, y_fit, test_size=0.33, random_state=seed, stratify=y_fit
    )
    model = clone(build_logreg_pipeline(X_tr)).fit(X_tr, y_tr)
    p_cal, p_te = model.predict_proba(X_cal), model.predict_proba(X_te)
    y_cal_arr, y_te_arr = y_cal.to_numpy(), y_te.to_numpy()
    sex_te = data.X.loc[X_te.index, "sex"].to_numpy()
    cal_scores = 1 - p_cal[np.arange(len(y_cal_arr)), y_cal_arr.astype(int)]

    records = []
    for alpha in ALPHAS:
        q_hat = conformal_quantile(cal_scores, alpha)
        sets_te = prediction_sets(p_te, q_hat)
        rec = {"alpha": alpha, "coverage": _coverage(sets_te, y_te_arr)}
        rec.update(_set_size_breakdown(sets_te))
        for sex in ("Female", "Male"):
            mask = sex_te == sex
            rec[f"coverage_{sex}"] = _coverage(sets_te[mask], y_te_arr[mask]) if mask.any() else np.nan
        records.append(rec)
    return records


def run(data) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_records = []
    for r in range(N_REPEATS):
        all_records.extend(_one_split(data, seed=RANDOM_SEED + r))
    df = pd.DataFrame(all_records)

    overall_rows, subgroup_rows = [], []
    for alpha in ALPHAS:
        sub = df[df["alpha"] == alpha]
        overall_rows.append(
            {
                "alpha": alpha,
                "target_coverage": round(1 - alpha, 3),
                "mean_coverage": round(sub["coverage"].mean(), 3),
                "coverage_std": round(sub["coverage"].std(), 3),
                "pct_splits_at_target": round((sub["coverage"] >= 1 - alpha).mean(), 3),
                "mean_set_size": round(sub["mean_set_size"].mean(), 3),
                "pct_uncertain": round(sub["pct_uncertain"].mean(), 3),
                "pct_empty": round(sub["pct_empty"].mean(), 3),
            }
        )
        for sex in ("Female", "Male"):
            subgroup_rows.append(
                {
                    "alpha": alpha,
                    "group": sex,
                    "target_coverage": round(1 - alpha, 3),
                    "mean_coverage": round(sub[f"coverage_{sex}"].mean(), 3),
                }
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(subgroup_rows)


def plot(overall: pd.DataFrame, subgroup: pd.DataFrame) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: target vs mean empirical coverage (error bars over splits).
    x = np.arange(len(overall))
    width = 0.35
    ax1.bar(x - width / 2, overall["target_coverage"], width, label="Target (1-α)", color="#bbbbbb")
    ax1.bar(
        x + width / 2, overall["mean_coverage"], width,
        yerr=overall["coverage_std"], capsize=4, label="Mean empirical", color="#4c72b0",
    )
    for i, r in overall.iterrows():
        ax1.annotate(
            f"uncertain {{0,1}}: {r['pct_uncertain']*100:.0f}%",
            (i, r["mean_coverage"]), xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=8,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"α={a}" for a in overall["alpha"]])
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("Coverage")
    ax1.set_title(f"Marginal coverage holds (mean of {N_REPEATS} splits)")
    ax1.legend(fontsize=8)

    # Right: conditional coverage by sex.
    groups = sorted(subgroup["group"].unique())
    xa = np.arange(len(ALPHAS))
    w = 0.35
    for j, g in enumerate(groups):
        sub = subgroup[subgroup["group"] == g].set_index("alpha").loc[ALPHAS]
        ax2.bar(xa + (j - 0.5) * w, sub["mean_coverage"], w, label=g)
    for i, a in enumerate(ALPHAS):
        ax2.hlines(1 - a, i - 0.5, i + 0.5, color="red", ls="--", lw=1)
    ax2.set_xticks(xa)
    ax2.set_xticklabels([f"α={a}" for a in ALPHAS])
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("Mean coverage")
    ax2.set_title("Conditional coverage by sex (red = target)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "conformal_coverage.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    data = load_cohort_data()
    overall, subgroup = run(data)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    overall.to_csv(os.path.join(REPORTS_DIR, "conformal_coverage.csv"), index=False)
    subgroup.to_csv(os.path.join(REPORTS_DIR, "conformal_coverage_by_sex.csv"), index=False)

    print(f"=== Split-conformal coverage (mean of {N_REPEATS} random splits) ===")
    print(overall.to_string(index=False))
    print("\n=== Conditional coverage by sex (marginal guarantee only) ===")
    print(subgroup.to_string(index=False))
    fig = plot(overall, subgroup)
    print(f"\nWrote: {fig}, reports/conformal_coverage.csv, reports/conformal_coverage_by_sex.csv")


if __name__ == "__main__":
    main()
