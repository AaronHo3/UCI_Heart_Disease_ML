"""Phase 5 — Missing data done properly.

Three questions:

1. **Is missingness informative — or is it just leaking the site?** Because
   ``ca``/``thal``/``slope`` are absent in whole cohorts, the missingness
   pattern nearly identifies the hospital, and the hospital determines
   prevalence. We show that (a) missingness indicators predict the *site* almost
   perfectly, (b) they look "predictive" of the outcome in a pooled split, and
   (c) that signal collapses under leave-one-site-out — i.e. it was leakage, the
   same trap as the ``dataset`` column in Phase 1.

2. **Does a smarter imputer help?** We compare median imputation (baseline),
   multivariate iterative imputation (MICE), and KNN imputation on pooled CV
   AUC and calibration. The honest expectation: little movement, because the
   heavily-missing features are nearly unrecoverable.

3. ``chol == 0`` is recoded to NaN upstream in :mod:`data` (a documented
   data-quality fix), so it is handled consistently everywhere here.

Outputs
-------
* ``reports/imputation_sensitivity.csv``
* ``reports/missingness_leakage.csv``
* ``reports/figures/imputation_sensitivity.png``
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (enables IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import load_cohort_data
from loso_validation import loso_predictions
from metrics import bootstrap_auc, expected_calibration_error
from pipeline_utils import RANDOM_SEED, build_hgb_pipeline, build_logreg_pipeline

REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
N_SPLITS = 5


def _numeric_imputer(kind: str):
    if kind == "median":
        return SimpleImputer(strategy="median")
    if kind == "mice":
        return IterativeImputer(random_state=RANDOM_SEED, max_iter=10, sample_posterior=False)
    if kind == "knn":
        return KNNImputer(n_neighbors=5)
    raise ValueError(f"unknown imputer {kind}")


def _preprocessor(X: pd.DataFrame, numeric_imputer_kind: str) -> ColumnTransformer:
    cat = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    num_pipe = Pipeline([("imputer", _numeric_imputer(numeric_imputer_kind)), ("scaler", StandardScaler())])
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)], remainder="drop")


def _logreg_with(numeric_imputer_kind: str):
    def builder(X: pd.DataFrame) -> Pipeline:
        return Pipeline(
            [
                ("prep", _preprocessor(X, numeric_imputer_kind)),
                ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_SEED)),
            ]
        )

    return builder


# --------------------------------------------------------------------------- #
# Missingness indicators
# --------------------------------------------------------------------------- #
def missingness_indicators(X: pd.DataFrame) -> pd.DataFrame:
    """Int 0/1 indicator per column that has any missing value."""
    cols = [c for c in X.columns if X[c].isna().any()]
    ind = X[cols].isna().astype(int)
    ind.columns = [f"{c}_missing" for c in cols]
    return ind


def _pooled_cv_auc(builder, X, y) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    proba = cross_val_predict(builder(X), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    auc = bootstrap_auc(np.asarray(y), proba)
    ece = expected_calibration_error(np.asarray(y), proba)
    return auc.value, ece


# --------------------------------------------------------------------------- #
# Q1. Is missingness informative or leaking the site?
# --------------------------------------------------------------------------- #
def missingness_leakage(data) -> pd.DataFrame:
    ind = missingness_indicators(data.X)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    rows = []

    # (a) Can missingness predict the SITE? (multiclass accuracy vs majority)
    site_clf = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))])
    acc = cross_val_score(site_clf, ind, data.site, cv=cv, scoring="accuracy").mean()
    majority = data.site.value_counts(normalize=True).max()
    rows.append({"test": "predict site from missingness (accuracy)", "value": round(acc, 3), "baseline": round(majority, 3)})

    # (b) Do missingness indicators ALONE "predict" the outcome (pooled)?
    ind_only = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))])
    proba = cross_val_predict(ind_only, ind, data.y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    rows.append(
        {"test": "predict outcome from missingness only, POOLED (AUC)", "value": round(bootstrap_auc(np.asarray(data.y), proba).value, 3), "baseline": 0.5}
    )

    # (c) Same within the one complete site (Cleveland): signal should vanish.
    cleveland = data.for_sites(["Cleveland"])
    cind = missingness_indicators(cleveland.X)
    if cind.shape[1] > 0 and cleveland.y.nunique() > 1:
        proba_c = cross_val_predict(
            Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))]),
            cind, cleveland.y, cv=cv, method="predict_proba", n_jobs=-1,
        )[:, 1]
        within = round(bootstrap_auc(np.asarray(cleveland.y), proba_c).value, 3)
    else:
        within = float("nan")
    rows.append({"test": "predict outcome from missingness only, WITHIN Cleveland (AUC)", "value": within, "baseline": 0.5})

    return pd.DataFrame(rows)


def _mean_per_site_auc(loso: pd.DataFrame) -> float:
    """Average of within-site held-out AUCs (prevalence proxies can't help here)."""
    aucs = []
    for _, sub in loso.groupby("site"):
        if sub["y_true"].nunique() > 1:
            aucs.append(bootstrap_auc(sub["y_true"].to_numpy(), sub["y_prob"].to_numpy()).value)
    return float(np.mean(aucs))


def indicator_feature_effect(data) -> pd.DataFrame:
    """Add missingness indicators as features and measure the effect three ways.

    - pooled_cv:   random-split ranking -> rewards a site/prevalence proxy (leak).
    - loso_pooled: ranks patients ACROSS held-out sites of differing prevalence,
      so it *also* rewards a prevalence proxy (a subtler leak in the metric).
    - loso_per_site: average WITHIN-site AUC. Indicators are ~constant within a
      site (missingness is site-determined), so honest transfer is unchanged.
    """
    from data import CohortData

    ind = missingness_indicators(data.X)
    X_aug = pd.concat([data.X.reset_index(drop=True), ind.reset_index(drop=True)], axis=1)

    base_pooled, _ = _pooled_cv_auc(build_logreg_pipeline, data.X, data.y)
    aug_pooled, _ = _pooled_cv_auc(build_logreg_pipeline, X_aug, data.y)

    base_loso = loso_predictions(build_logreg_pipeline, data)
    aug_loso = loso_predictions(build_logreg_pipeline, CohortData(X=X_aug, y=data.y, site=data.site))

    base_lp = bootstrap_auc(base_loso["y_true"].to_numpy(), base_loso["y_prob"].to_numpy()).value
    aug_lp = bootstrap_auc(aug_loso["y_true"].to_numpy(), aug_loso["y_prob"].to_numpy()).value
    base_ps = _mean_per_site_auc(base_loso)
    aug_ps = _mean_per_site_auc(aug_loso)

    def row(regime, b, a):
        return {"regime": regime, "without_indicators": round(b, 4), "with_indicators": round(a, 4), "delta": round(a - b, 4)}

    return pd.DataFrame(
        [
            row("pooled_cv", base_pooled, aug_pooled),
            row("loso_pooled", base_lp, aug_lp),
            row("loso_per_site_mean", base_ps, aug_ps),
        ]
    )


# --------------------------------------------------------------------------- #
# Q2. Imputation sensitivity
# --------------------------------------------------------------------------- #
def imputation_sensitivity(data) -> pd.DataFrame:
    rows = []
    for imputer in ("median", "mice", "knn"):
        auc_lr, ece_lr = _pooled_cv_auc(_logreg_with(imputer), data.X, data.y)
        rows.append({"model": "Logistic Regression", "imputer": imputer, "auc": round(auc_lr, 4), "ece": round(ece_lr, 4)})
    # Gradient Boosting handles NaNs natively, so it is the natural "no imputation" comparator.
    auc_gb, ece_gb = _pooled_cv_auc(build_hgb_pipeline, data.X, data.y)
    rows.append({"model": "Gradient Boosting", "imputer": "native (no impute)", "auc": round(auc_gb, 4), "ece": round(ece_gb, 4)})
    return pd.DataFrame(rows)


def plot_imputation(table: pd.DataFrame) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    lr = table[table["model"] == "Logistic Regression"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(lr))
    ax.bar(x, lr["auc"], width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(lr["imputer"])
    ax.set_ylim(0.80, 0.90)
    ax.set_ylabel("Pooled CV ROC-AUC")
    ax.set_title("Imputation sensitivity (Logistic Regression)")
    for i, v in enumerate(lr["auc"]):
        ax.annotate(f"{v:.3f}", (i, v), ha="center", xytext=(0, 3), textcoords="offset points")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "imputation_sensitivity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    data = load_cohort_data()
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=== Q1a/b/c: is missingness informative, or leaking the site? ===")
    leak = missingness_leakage(data)
    print(leak.to_string(index=False))

    print("\n=== Q1d: indicator features — pooled (leaky) vs LOSO (honest) ===")
    eff = indicator_feature_effect(data)
    print(eff.to_string(index=False))
    eff.to_csv(os.path.join(REPORTS_DIR, "missingness_leakage.csv"), index=False)
    leak.to_csv(os.path.join(REPORTS_DIR, "missingness_predictiveness.csv"), index=False)

    print("\n=== Q2: imputation sensitivity (pooled CV) ===")
    sens = imputation_sensitivity(data)
    print(sens.to_string(index=False))
    sens.to_csv(os.path.join(REPORTS_DIR, "imputation_sensitivity.csv"), index=False)
    fig = plot_imputation(sens)
    print(f"\nWrote: {fig} and three CSVs in reports/")


if __name__ == "__main__":
    main()
