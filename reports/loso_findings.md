# Phase 1 — Leave-One-Site-Out External Validation

**Research question:** Does a heart-disease classifier trained on some hospitals
still work on a *different* hospital it has never seen?

Reproduce with `make loso` (writes `reports/loso_results.csv` and the figures
below). Seed: `RANDOM_SEED = 100`. Metrics carry 95% bootstrap CIs
(`n_boot = 2000`).

## Why this matters

The four cohorts differ dramatically — not just in feature distributions but in
**outcome prevalence** and **feature availability**:

| Cohort | n | Prevalence (num>0) | % Male | Notable missingness |
|---|---:|---:|---:|---|
| Hungary | 293 | 36.2% | 72% | slope 65%, ca 99%, thal 90% |
| Cleveland | 304 | 45.7% | 68% | ~complete |
| VA Long Beach | 200 | 74.5% | 97% | vitals ~28%, ca 99%, thal 83% |
| Switzerland | 123 | **93.5%** (8 negatives) | 92% | chol 100%=0, fbs 61%, ca 96% |

Three stacked shifts: **label shift** (prevalence 36%→93%), **covariate shift**
(sex/age/vitals), and **feature-support shift** (`ca`/`thal`/`slope` are
effectively Cleveland-only).

## Result 1 — Generalization gap

A pooled random-split CV (the number most portfolio projects report) is
optimistic. Holding out an entire hospital reveals the real drop:

| Model | Pooled-CV AUC | LOSO AUC | Gap |
|---|---:|---:|---:|
| Logistic Regression | 0.879 | 0.821 | **−0.058** |
| Random Forest | 0.876 | 0.801 | **−0.075** |
| Gradient Boosting | 0.866 | 0.785 | **−0.081** |

The simplest model generalizes best; flexible learners overfit site-specific
structure. → `reports/figures/loso_gap.png`

## Result 2 — Per-site discrimination is uneven and sometimes unmeasurable

Per-site held-out AUC (Gradient Boosting):

| Site | n | n_pos | AUC [95% CI] | ECE |
|---|---:|---:|---|---:|
| Cleveland | 304 | 139 | 0.81 [0.76, 0.86] | 0.14 |
| Hungary | 293 | 106 | 0.84 [0.79, 0.88] | 0.15 |
| Switzerland | 123 | 115 | 0.79 [**0.61, 0.96**] | 0.30 |
| VA Long Beach | 200 | 149 | **0.66 [0.56, 0.76]** | 0.22 |

Switzerland's interval is enormous: with only 8 negatives its AUC is not a
trustworthy discrimination estimate, and the CI says so. VA Long Beach is the
genuine failure — the model barely ranks better than chance there.
→ `reports/figures/loso_per_site.png`

## Result 3 — Calibration collapses under label shift

Mean predicted probability barely moves (≈0.47→0.64) while true prevalence
spans 0.36→0.94. Every site sits below the calibration diagonal: the model
exports its ~50% training prevalence and cannot adapt. ECE rises to 0.30
(Switzerland) and 0.22 (VA). → `reports/figures/loso_calibration_drift.png`
This motivates the recalibration + decision-curve analysis in Phase 3.

## Result 4 — The dropped `dataset` column leaks prevalence (now empirical)

Re-adding the site indicator as a feature inflates pooled-CV AUC by **+0.025**
(0.879 → 0.904 for logistic regression). Because prevalence ranges 36%→93%
across sites, a model that can see the site shortcuts to its base rate. This is
direct evidence — not assertion — that the site column must stay out of the
features.

## Honest limitations

- Switzerland and VA have very few negatives / heavy missingness; their per-site
  numbers are reported with wide CIs and should not be over-interpreted.
- `ca`/`thal`/`slope` are largely absent outside Cleveland, so part of the
  generalization gap is feature-support shift, not just covariate/label shift.
  Phase 5 examines this directly.
