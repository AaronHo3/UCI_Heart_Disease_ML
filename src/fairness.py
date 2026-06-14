"""Phase 7 - Fairness / subgroup audit.

Aggregate AUC hides who the model fails. Heart disease is historically
under-diagnosed in women, and this dataset skews heavily male (and the
high-prevalence cohorts are 92-97% male), so a sex-disaggregated audit is
essential - reported with subgroup sizes so thin groups are not over-read.

For each subgroup we report n, prevalence, AUC (95% CI), calibration (ECE), and
the two error rates that matter clinically at a 0.5 operating point:

* **FNR** (false-negative rate = missed disease) - the equity-critical metric;
* **FPR** (false-positive rate = unnecessary work-up).

Outputs
-------
* ``reports/fairness_subgroups.csv``
* ``reports/figures/fairness_by_sex.png``, ``fairness_by_age.png``
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_cohort_data
from loso_validation import pooled_cv_predictions
from metrics import bootstrap_auc, brier, expected_calibration_error
from pipeline_utils import build_logreg_pipeline

REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
THRESHOLD = 0.5
AGE_BINS = [0, 50, 60, 200]
AGE_LABELS = ["<50", "50-59", "60+"]


def _error_rates(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> tuple[float, float]:
    pred = (y_prob >= threshold).astype(int)
    tp = np.sum((pred == 1) & (y_true == 1))
    fn = np.sum((pred == 0) & (y_true == 1))
    tn = np.sum((pred == 0) & (y_true == 0))
    fp = np.sum((pred == 1) & (y_true == 0))
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return float(fnr), float(fpr)


def _subgroup_row(name: str, value: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    auc = bootstrap_auc(y_true, y_prob)
    fnr, fpr = _error_rates(y_true, y_prob, THRESHOLD)
    return {
        "attribute": name,
        "group": value,
        "n": len(y_true),
        "n_pos": int(y_true.sum()),
        "prevalence": round(float(y_true.mean()), 3),
        "auc": round(auc.value, 3),
        "auc_lo": round(auc.lower, 3),
        "auc_hi": round(auc.upper, 3),
        "ece": round(expected_calibration_error(y_true, y_prob), 3),
        "brier": round(brier(y_true, y_prob), 3),
        "fnr": round(fnr, 3),
        "fpr": round(fpr, 3),
    }


def run(data) -> pd.DataFrame:
    proba = pooled_cv_predictions(build_logreg_pipeline(data.X), data.X, data.y)
    y = data.y.to_numpy()
    frame = pd.DataFrame(
        {
            "y": y,
            "proba": proba,
            "sex": data.X["sex"].to_numpy(),
            "age_band": pd.cut(data.X["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False),
        }
    )

    rows = [_subgroup_row("overall", "all", frame["y"].to_numpy(), frame["proba"].to_numpy())]
    for sex in sorted(frame["sex"].dropna().unique()):
        sub = frame[frame["sex"] == sex]
        rows.append(_subgroup_row("sex", str(sex), sub["y"].to_numpy(), sub["proba"].to_numpy()))
    for band in AGE_LABELS:
        sub = frame[frame["age_band"] == band]
        if len(sub) > 0:
            rows.append(_subgroup_row("age", band, sub["y"].to_numpy(), sub["proba"].to_numpy()))
    return pd.DataFrame(rows)


def plot_by_sex(table: pd.DataFrame) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sex = table[table["attribute"] == "sex"].reset_index(drop=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(sex))
    vals = sex["auc"].to_numpy()
    err = [vals - sex["auc_lo"].to_numpy(), sex["auc_hi"].to_numpy() - vals]
    ax1.bar(x, vals, yerr=err, capsize=4, color=["#4c72b0", "#dd8452"])
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{r.group}\n(n={r.n}, pos={r.n_pos})" for r in sex.itertuples()])
    ax1.set_ylim(0.5, 1.0)
    ax1.set_ylabel("AUC (95% CI)")
    ax1.set_title("Discrimination by sex")

    width = 0.35
    ax2.bar(x - width / 2, sex["fnr"], width, label="FNR (missed disease)", color="#c44e52")
    ax2.bar(x + width / 2, sex["fpr"], width, label="FPR (false alarms)", color="#8172b3")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sex["group"])
    ax2.set_ylabel(f"Error rate @ threshold {THRESHOLD}")
    ax2.set_title("Error-rate gaps by sex")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fairness_by_sex.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_by_age(table: pd.DataFrame) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    age = table[table["attribute"] == "age"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(age))
    vals = age["auc"].to_numpy()
    err = [vals - age["auc_lo"].to_numpy(), age["auc_hi"].to_numpy() - vals]
    ax.bar(x, vals, yerr=err, capsize=4, color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.group}\n(n={r.n}, prev={r.prevalence})" for r in age.itertuples()])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC (95% CI)")
    ax.set_title("Discrimination by age band")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fairness_by_age.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    data = load_cohort_data()
    table = run(data)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    table.to_csv(os.path.join(REPORTS_DIR, "fairness_subgroups.csv"), index=False)

    print("=== Subgroup audit (pooled CV, logistic regression) ===")
    print(table.to_string(index=False))

    sex = table[table["attribute"] == "sex"].set_index("group")
    if {"Male", "Female"}.issubset(sex.index):
        gap = sex.loc["Female", "fnr"] - sex.loc["Male", "fnr"]
        print(f"\nFNR gap (Female - Male): {gap:+.3f}  (positive = women's disease missed more often)")

    paths = [plot_by_sex(table), plot_by_age(table)]
    print("\nWrote:", ", ".join(paths), "and reports/fairness_subgroups.csv")


if __name__ == "__main__":
    main()
