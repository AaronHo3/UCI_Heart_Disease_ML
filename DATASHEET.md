# Datasheet - UCI Heart Disease (4-cohort aggregate)

Following Gebru et al., *Datasheets for Datasets* (2021).

## Motivation
- **Purpose:** study coronary-artery-disease prediction from routine clinical
  and exercise-test measurements. Used here to study clinical-ML *methodology*.
- **Created by:** Hungarian Institute of Cardiology (Budapest); University
  Hospitals Zurich & Basel (Switzerland); V.A. Medical Center, Long Beach;
  Cleveland Clinic Foundation. Distributed via the UCI ML Repository.

## Composition
- **Instances:** 920 patients, each a single clinical encounter, from **four
  hospital cohorts** pooled together (`dataset` column).
- **Label:** `num` (0-4 angiographic disease severity); used as binary `num > 0`.
  Pooled prevalence 55.3%; per-cohort 36.2%-93.5%.
- **Features (13):** age, sex, chest-pain type (cp), resting BP (trestbps),
  cholesterol (chol), fasting blood sugar (fbs), resting ECG (restecg), max heart
  rate (thalch), exercise angina (exang), ST depression (oldpeak), ST slope
  (slope), number of major vessels (ca), thalassemia (thal).
- **Missingness is structured and cohort-dependent:** `ca`/`thal`/`slope` are
  ~90-99% missing outside Cleveland; Switzerland has 61% `fbs` missing.

## Known data-quality issues
- **`chol == 0`** is physiologically impossible - it encodes *missing* serum
  cholesterol. It is the entire Switzerland cohort (123/123) plus 49 VA patients.
  This project recodes it to NaN in `src/data.py`.
- **Class imbalance varies wildly by site** (label shift); Switzerland has only
  8 negative cases.
- **Demographic skew:** 79% male overall; high-prevalence cohorts are 92-97%
  male, limiting subgroup power for women.

## Collection process
- Retrospective clinical data with invasive angiography as the disease reference.
  Decades old; collection protocols differ by site, which is the source of the
  feature-support shift.

## Preprocessing / labeling
- Binary target `num > 0`; `id`/`num`/`dataset` excluded from features (`dataset`
  retained only as a grouping variable). `chol == 0` -> NaN. Numeric features
  median-imputed + scaled; categoricals mode-imputed + one-hot encoded.

## Uses
- **Used here for:** external-validation, calibration, fairness, and
  uncertainty-quantification methodology.
- **Should not be used for:** real diagnosis or any clinical deployment. Findings
  do not transfer to modern populations or other hospitals.

## Distribution & maintenance
- Source: UCI ML Repository / Kaggle (`redwankarimsony/heart-disease-data`).
  Citation in `CITATION.md`. The CSV is committed under `data/raw/` for
  reproducibility; `make download` can re-fetch it.
