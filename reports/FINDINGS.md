# Findings

A single, living record of results as the study progresses. Every number is
reproducible from a clean clone on seed `RANDOM_SEED = 100`. Metrics carry 95%
bootstrap confidence intervals (`n_boot = 2000`) unless noted.

| Phase | Topic | Reproduce |
|---|---|---|
| 1 | Leave-one-site-out external validation | `make loso` |
| 2 | Nested CV, bootstrap CIs, DeLong tests | `make nested` |
| 3 | Calibration, decision curves, clinical baseline | `make calibrate` / `make dca` |
| 5 | Missing data: leakage + imputation sensitivity | `make missingness` |

---

## Dataset: four hospital cohorts, three stacked shifts

The UCI aggregate (n=920) bundles four cohorts that differ not only in feature
distributions but in **outcome prevalence** and **feature availability**:

| Cohort | n | Prevalence (num>0) | % Male | Notable missingness |
|---|---:|---:|---:|---|
| Hungary | 293 | 36.2% | 72% | slope 65%, ca 99%, thal 90% |
| Cleveland | 304 | 45.7% | 68% | ~complete |
| VA Long Beach | 200 | 74.5% | 97% | vitals ~28%, ca 99%, thal 83% |
| Switzerland | 123 | **93.5%** (8 negatives) | 92% | chol 100%=0, fbs 61%, ca 96% |

- **Label shift** (dominant): prevalence spans 36%→93%.
- **Covariate shift**: sex/age/vitals distributions differ by site.
- **Feature-support shift**: `ca`/`thal`/`slope` (classically the most
  predictive features) are ~90–99% missing outside Cleveland.

---

## Phase 1 — Leave-One-Site-Out (LOSO) external validation

**Question:** does a model trained on some hospitals work on a *different* one?

### Result 1 — A real, consistent generalization gap

Pooled random-split CV (what most portfolio projects report) is optimistic;
holding out an entire hospital reveals the real drop:

| Model | Pooled-CV AUC | LOSO AUC | Gap |
|---|---:|---:|---:|
| Logistic Regression | 0.879 | 0.821 | **−0.058** |
| Random Forest | 0.876 | 0.801 | **−0.075** |
| Gradient Boosting | 0.866 | 0.785 | **−0.081** |

The simplest model transfers best; flexible learners overfit site-specific
structure. → `figures/loso_gap.png`

### Result 2 — Per-site discrimination is uneven, sometimes unmeasurable

Gradient Boosting, held-out per site:

| Site | n | n_pos | AUC [95% CI] | ECE |
|---|---:|---:|---|---:|
| Cleveland | 304 | 139 | 0.81 [0.76, 0.86] | 0.14 |
| Hungary | 293 | 106 | 0.84 [0.79, 0.88] | 0.15 |
| Switzerland | 123 | 115 | 0.79 [**0.61, 0.96**] | 0.30 |
| VA Long Beach | 200 | 149 | **0.66 [0.56, 0.76]** | 0.22 |

Switzerland's interval is enormous (only 8 negatives) — its AUC is not a
trustworthy discrimination estimate, and the CI says so. VA Long Beach is the
genuine failure: barely better than chance. → `figures/loso_per_site.png`

### Result 3 — Calibration collapses under label shift

Mean predicted probability barely moves (≈0.47→0.64) while true prevalence
spans 0.36→0.94; every site sits below the calibration diagonal. The model
exports its ~50% training prevalence and cannot adapt. ECE rises to 0.30
(Switzerland), 0.22 (VA). → `figures/loso_calibration_drift.png`

### Result 4 — The dropped `dataset` column leaks prevalence (now empirical)

Re-adding the site indicator as a feature inflates pooled-CV AUC by **+0.025**
(0.879 → 0.904, logistic regression). A model that can see the site shortcuts to
its base rate — direct evidence (not assertion) that the site column must stay
out of the features.

**Limitations:** Switzerland/VA have few negatives and heavy missingness — wide
CIs, do not over-interpret. Part of the gap is feature-support shift
(`ca`/`thal`/`slope` largely absent outside Cleveland); examined in Phase 5.

---

## Phase 2 — Nested CV, bootstrap CIs, and DeLong tests

### Headline metrics (nested CV, 95% bootstrap CI)

| Model | ROC-AUC | Accuracy | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Logistic Regression | 0.880 [0.858, 0.902] | — | — | — | — |
| Random Forest | 0.875 [0.852, 0.897] | — | — | — | — |
| Gradient Boosting | 0.880 [0.858, 0.901] | — | — | — | — |

(Full accuracy/F1/sensitivity/specificity with CIs in
`reports/nested_cv_results.csv`.)

### Optimism bias — why nested CV matters

Tuning hyperparameters on the same folds you then report inflates AUC:

| Model | Naive CV AUC | Nested CV AUC | Optimism |
|---|---:|---:|---:|
| Logistic Regression | 0.883 | 0.880 | +0.002 |
| Random Forest | 0.884 | 0.875 | +0.009 |
| Gradient Boosting | 0.886 | 0.880 | +0.006 |

The bias is largest for the most flexible model (Random Forest). The nested
estimates are the honest ones. → `figures/nested_vs_naive_auc.png`

### Are the models actually different? (DeLong test)

| Comparison | AUC A | AUC B | Δ | p-value | Verdict |
|---|---:|---:|---:|---:|---|
| LogReg vs Random Forest | 0.880 | 0.875 | +0.005 | 0.352 | **indistinguishable** |
| LogReg vs Gradient Boosting | 0.880 | 0.880 | +0.000 | 0.961 | **indistinguishable** |
| Random Forest vs Gradient Boosting | 0.875 | 0.880 | −0.005 | 0.331 | **indistinguishable** |

On this well-trodden dataset the three model families are statistically
indistinguishable by AUC. Combined with Phase 1 (logistic regression transfers
best and is the most interpretable), model choice should be driven by
calibration, robustness, and interpretability — not a non-significant AUC gap.

---

## Phase 3 — Calibration, decision curves, and a clinical baseline

Evaluated on a held-out 20% test split (seed 100); the test set is never used
for fitting or recalibration.

### Calibration — does "0.8" mean an 80% chance of disease?

In-distribution (pooled) calibration, Brier / ECE by recalibration method:

| Model | Uncalibrated (Brier/ECE) | Sigmoid | Isotonic |
|---|---|---|---|
| Logistic Regression | 0.130 / 0.052 | 0.131 / 0.066 | 0.137 / 0.101 |
| Random Forest | 0.127 / 0.065 | 0.129 / 0.101 | 0.130 / 0.097 |
| Gradient Boosting | 0.147 / **0.106** | 0.135 / 0.067 | 0.135 / **0.042** |

Logistic regression is well-calibrated out of the box; recalibrating it only
adds noise. Gradient Boosting is the worst calibrated but **isotonic
recalibration more than halves its ECE** (0.106 → 0.042). → `figures/calibration_reliability.png`

Note this is the *in-distribution* picture. Phase 1 showed calibration breaks
badly *across sites* (ECE up to 0.30) — a harder problem that simple
recalibration on the source sites cannot fix because it is label shift, not
just miscalibration.

### Decision Curve Analysis — is the model clinically useful?

Across the clinically relevant threshold range both the full model and the
clinical proxy yield **higher net benefit than treat-all and treat-none**, so
acting on the model beats the trivial policies. → `figures/decision_curve.png`

### Standard-of-care baseline — does complexity actually help?

A deliberately simple **clinical risk proxy** (logistic regression on 7
routinely-available history/exercise variables: age, sex, cp, trestbps, thalch,
exang, oldpeak) vs the full 13-feature Gradient Boosting model:

| Model | # features | AUC [95% CI] |
|---|---:|---|
| Full ML (GB, all features) | 13 | 0.874 [0.822, 0.922] |
| Clinical proxy (LogReg, routine features) | 7 | 0.881 [0.831, 0.924] |

DeLong: Δ = −0.007, **p = 0.68 → indistinguishable**. The complex model does
**not** beat a simple, deployable clinical baseline. (This proxy is *not* a
validated score like Framingham/ASCVD — those require HDL, smoking,
BP-treatment and diabetes data this dataset lacks — it is an honest
"is the complexity worth it?" comparator.)

---

## Phase 5 — Missing data done properly

`chol == 0` (physiologically impossible — the entire Switzerland cohort plus
part of VA) is recoded to NaN in `src/data.py` so it is handled honestly
everywhere.

### Is missingness informative, or is it leaking the site?

`ca`/`thal`/`slope` are absent in whole cohorts, so the missingness *pattern*
nearly identifies the hospital:

| Test | Value | Baseline |
|---|---:|---:|
| Predict **site** from missingness indicators (accuracy) | **0.855** | 0.33 |
| Predict outcome from missingness only — **pooled** (AUC) | 0.747 | 0.50 |
| Predict outcome from missingness only — **within Cleveland** (AUC) | **0.498** | 0.50 |

Missingness "predicts" disease at AUC 0.747 across the pooled data, but **0.498
(pure chance) within the one complete cohort**. The apparent signal is entirely
a site→prevalence proxy, not clinical information.

### Adding missingness indicators as features — measured three ways

| Evaluation | Without indicators | With indicators | Δ |
|---|---:|---:|---:|
| Pooled random-split CV | 0.879 | 0.900 | **+0.021** |
| LOSO, pooled across sites | 0.821 | 0.864 | **+0.043** |
| LOSO, **mean within-site** | 0.802 | 0.808 | **+0.006** |

Indicators help only when the metric ranks patients *across* populations of
different prevalence (pooled CV, and even pooled-LOSO). The honest **within-site**
transfer is essentially unchanged (+0.006 ≈ noise). Two lessons: (1) missingness
here is a prevalence proxy, not signal; (2) even "external validation" can leak
prevalence if you pool across sites — per-site evaluation is the trustworthy one.

### Does a smarter imputer help?

Pooled CV (logistic regression), cholesterol and the rest imputed by:

| Imputer | AUC | ECE |
|---|---:|---:|
| Median (baseline) | 0.879 | 0.029 |
| MICE (IterativeImputer) | 0.886 | 0.033 |
| KNN | 0.882 | 0.025 |
| Gradient Boosting (native NaN handling) | 0.866 | 0.097 |

Imputation strategy barely moves AUC (all within each other's bootstrap CI);
MICE gives a tiny, non-significant bump. Fancy imputation cannot recover
features that are 90–99% absent — consistent with the feature-support shift
identified in Phase 1. → `figures/imputation_sensitivity.png`
