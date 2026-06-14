# Findings

A single, living record of results as the study progresses. Every number is
reproducible from a clean clone on seed `RANDOM_SEED = 100`. Metrics carry 95%
bootstrap confidence intervals (`n_boot = 2000`) unless noted.

| Phase | Topic | Reproduce |
|---|---|---|
| 1 | Leave-one-site-out external validation | `make loso` |
| 2 | Nested CV, bootstrap CIs, DeLong tests | `make nested` |

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
