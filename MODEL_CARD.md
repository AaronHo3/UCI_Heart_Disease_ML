# Model Card - Heart Disease Classifier

Following Mitchell et al., *Model Cards for Model Reporting* (2019). This card
describes the models studied in this repository. **These models are research
artifacts for methodology demonstration, not medical devices.**

## Model details
- **Developers:** Aaron Ho (portfolio / research project).
- **Models:** scikit-learn pipelines (median/mode impute -> scale -> one-hot ->
  classifier) for (a) logistic regression, (b) random forest, (c)
  HistGradientBoosting. Logistic regression is the **recommended** model.
- **Version / provenance:** seed 100; exact commit, library versions, and
  artifact hashes recorded in `reports/run_manifest.json`.
- **Task:** binary classification of heart disease (`num > 0`) from 13 clinical
  features.

## Intended use
- **Intended:** education and research on clinical-ML methodology - external
  validation, calibration, fairness, uncertainty quantification.
- **Out of scope:** real clinical decision-making, diagnosis, triage, or any
  deployment affecting patients. The data is decades old, from four specific
  hospitals, and small.

## Factors
- Evaluated across **hospital site** (Cleveland, Hungary, Switzerland, VA),
  **sex**, and **age band**. These materially change performance.

## Metrics
- **Discrimination:** ROC-AUC with 95% bootstrap CIs; DeLong tests for model
  comparison.
- **Calibration:** Brier score, Expected Calibration Error, reliability curves.
- **Clinical utility:** decision-curve net benefit.
- **Error rates:** sensitivity/specificity, FNR/FPR by subgroup.
- **Uncertainty:** split-conformal coverage and prediction-set size.

## Quantitative analysis (headline)
- Pooled-CV AUC ~0.88; **mean AUC within a held-out hospital 0.78-0.80** (the
  honest estimate; pooling all out-of-site predictions gives a milder 0.79-0.82
  but rewards cross-cohort prevalence ranking). Models are statistically
  indistinguishable (every pairwise DeLong p far above 0.05).
- A 7-feature clinical proxy is indistinguishable from the full model (p = 0.68).
- Full numbers with CIs: `reports/FINDINGS.md`. Narrative: `REPORT.md`.

## Ethical considerations & known limitations
- **Fairness:** at a single 0.5 threshold the model misses women's disease ~2.4x
  as often as men's (FNR 0.36 vs 0.15), an artifact of label shift / global
  thresholding. Mitigation: group-aware, prevalence-adjusted thresholds.
- **Generalization:** performance degrades on unseen hospitals; calibration
  breaks under cross-site label shift.
- **Data quality:** `chol == 0` is a missingness artifact (recoded); outside
  Cleveland `ca` is 98% missing, `thal` 78%, and `slope` 50%.
- **Reproducibility:** logistic-regression and gradient-boosting results
  reproduce bit-exactly on the pinned versions; Random Forest drifts in the
  third decimal across platforms.
- **Uncertainty:** conformal guarantees marginal, not conditional, coverage.

## Recommendation
If used at all (for research), use **logistic regression with group-aware
thresholds and conformal abstention**, and report site-specific, subgroup-specific
metrics with confidence intervals - never a single pooled number.
