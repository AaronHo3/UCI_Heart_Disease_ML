# Does Heart-Disease Prediction Transfer Across Hospitals?
### A multi-site external-validation study on the UCI Heart Disease dataset

**TL;DR.** On a famous, well-trodden dataset, the prediction itself is not the
interesting part. This report reframes it as a *clinical machine-learning
methodology* study: how well does a heart-disease classifier trained on some
hospitals generalize to a hospital it has never seen, and is its output
trustworthy enough to act on? The headline result is that pooled cross-validation
(AUC ~0.88) substantially overstates real transfer (leave-one-site-out AUC ~
0.78-0.82), that probability calibration breaks under cross-site label shift, and
that a 7-feature clinical baseline is statistically indistinguishable from the
full model. Every claim is reported with a confidence interval and, where
comparative, a statistical test.

---

## 1. Research questions

1. **Generalization.** Does performance estimated by pooled cross-validation
   survive when an entire hospital is held out (out-of-distribution)?
2. **Honesty of the numbers.** After removing optimistic bias (nested CV) and
   attaching uncertainty (bootstrap CIs, DeLong tests), are the models actually
   different, and how good are they really?
3. **Trustworthiness.** Are the predicted probabilities calibrated, clinically
   useful (decision-curve net benefit), and better than a simple standard-of-care
   proxy?
4. **Mechanism & fairness.** What drives the model, does it agree with
   cardiology, how does it fail across patient subgroups, and can it quantify its
   own uncertainty?

## 2. Data

UCI Heart Disease (n = 920), an aggregate of four hospital cohorts. Binary target
`num > 0`. The cohorts differ on three axes simultaneously - this heterogeneity
*is* the study:

| Cohort | n | Prevalence | % Male | Notable missingness |
|---|---:|---:|---:|---|
| Hungary | 293 | 36.2% | 72% | slope 65%, ca 99%, thal 90% |
| Cleveland | 304 | 45.7% | 68% | ~complete |
| VA Long Beach | 200 | 74.5% | 97% | vitals ~28%, ca 99%, thal 83% |
| Switzerland | 123 | 93.5% (8 neg.) | 92% | chol 100%=0, fbs 61%, ca 96% |

- **Label shift**: prevalence 36% -> 93.5%.
- **Covariate shift**: sex/age/vitals.
- **Feature-support shift**: `ca`/`thal`/`slope` are effectively Cleveland-only.

See the [Datasheet](DATASHEET.md) for provenance and known artifacts (e.g.
`chol == 0` recoded to missing).

## 3. Methods

scikit-learn pipelines (impute -> scale -> one-hot -> classifier) for logistic
regression, random forest, and gradient boosting (HistGradientBoosting), seed
100. Specific techniques:

- **LOSO external validation** (train on 3 cohorts, test on the 4th) vs pooled
  stratified CV.
- **Nested CV** for unbiased model selection; **bootstrap** 95% CIs;
  **DeLong's test** for AUC differences.
- **Calibration**: reliability curves, Brier, ECE, Platt & isotonic recalibration.
- **Decision-curve analysis**; a parsimonious **clinical risk proxy** baseline.
- **SHAP**, permutation importance, partial dependence; **clinical cross-check**.
- **Missingness** leakage analysis + imputation sensitivity (median/MICE/KNN).
- **Subgroup fairness** audit (sex, age); **split-conformal** prediction sets.

Full numeric results with CIs: [reports/FINDINGS.md](reports/FINDINGS.md). Every
analysis is one `make` target; provenance in `reports/run_manifest.json`.

## 4. Results

**(1) The generalization gap is real.** Pooled CV -> LOSO AUC drops -0.058
(LogReg), -0.075 (RF), -0.081 (GB). The simplest model transfers best.
`reports/figures/loso_gap.png`

**(2) Per-site failure is uneven.** VA Long Beach AUC collapses to 0.66
[0.56, 0.76]; Switzerland is uninterpretable (8 negatives -> 0.79 [0.61, 0.96]).
`reports/figures/loso_per_site.png`

**(3) Calibration breaks under label shift.** Predicted probabilities barely move
(~0.47->0.64) while true prevalence spans 0.36->0.94; ECE rises to 0.30.
`reports/figures/loso_calibration_drift.png`

**(4) The models are statistically indistinguishable.** Nested-CV AUCs 0.880 /
0.875 / 0.880; all pairwise DeLong tests p > 0.33. Optimism bias (naive - nested)
is largest for the most flexible model. `reports/figures/nested_vs_naive_auc.png`

**(5) Complexity does not beat a simple baseline.** A 7-feature clinical proxy
(AUC 0.881 [0.83, 0.92]) is indistinguishable from the 13-feature GB model
(0.874 [0.82, 0.92]), DeLong p = 0.68. On net benefit both beat treat-none
across the range, and both beat treat-all above a threshold of 0.14 (0.08 for
the proxy); below that, treat-all is as good or better.
`reports/figures/decision_curve.png`

**(6) The model has clinical face validity.** 8/8 salient features
(oldpeak, ca, exang, thalch, age, sex, cp, thal) are learned in the
cardiologically correct direction. `reports/figures/shap_summary.png`

**(7) Missingness is a site/prevalence proxy, not signal.** It predicts the
*site* at 85.5% accuracy and "predicts" disease at AUC 0.747 pooled but 0.498
(chance) within Cleveland. Adding indicators inflates pooled AUC but leaves mean
within-site AUC unchanged (+0.006).

**(8) A fairness gap hides behind equal AUC.** At a single 0.5 threshold the
model misses women's disease 2.4x as often as men's (FNR 0.36 vs 0.15), driven by
the same label-shift/calibration mechanism. `reports/figures/fairness_by_sex.png`

**(9) Honest uncertainty.** Split-conformal prediction achieves its target
coverage (0.908 at 90%) and abstains (`{0,1}`) on ~27% of patients - an
actionable deferral signal. `reports/figures/conformal_coverage.png`

## 5. Discussion

The recurring theme is that a single number (pooled AUC, a global threshold) is
optimistic and can hide real problems: degraded transfer, miscalibration, and
inequitable error rates. Three independent analyses (LOSO, the cohort/missingness
leakage probes, the subgroup audit) all trace back to **label shift across
sites**. The practical recommendation is **logistic regression with group-aware,
prevalence-adjusted thresholds and conformal abstention** - it transfers best, is
interpretable, is statistically as accurate as anything more complex, and its
uncertainty can be quantified with a guarantee.

## 6. Limitations

- Switzerland/VA have few negatives and heavy missingness; their per-site numbers
  carry wide CIs and should not be over-interpreted.
- The "clinical proxy" is **not** a validated score (the data lacks Framingham/
  ASCVD inputs); it is an honest "is the complexity worth it?" comparator.
- Conformal guarantees marginal, not conditional, coverage.
- n = 920 across four cohorts limits subgroup power, especially for women.

## 7. What I would do next

- Group-conditional (Mondrian) conformal and group-aware thresholds, then
  re-audit the FNR gap.
- Domain-adaptation / label-shift correction (e.g. prevalence re-weighting) to
  recover cross-site calibration.
- A prospective, multi-site dataset with consistent feature collection to remove
  the feature-support shift that bounds transfer here.

## 8. Reproducibility

```bash
make setup            # create venv, install requirements
make download         # fetch the dataset
make loso nested calibrate dca interpret missingness fairness conformal
make manifest         # record provenance -> reports/run_manifest.json
pytest -q             # 27 unit/smoke tests; CI runs lint + tests
```

Seed 100 throughout. See the [Model Card](MODEL_CARD.md) for intended use and
the [Datasheet](DATASHEET.md) for data provenance.
