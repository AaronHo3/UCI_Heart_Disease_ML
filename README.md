# Heart Disease Classification (Logistic Regression + Random Forest + Gradient Boosting)

A reproducible machine learning pipeline for predicting heart disease from clinical features using logistic regression, random forest, and gradient boosting models.

This project demonstrates an end-to-end ML workflow including preprocessing, cross-validation, model comparison, evaluation, interpretability, and reproducibility practices.

---

## Overview

The goal is to predict whether a patient has heart disease (`num > 0`) using demographic, clinical, and diagnostic features from the UCI Heart Disease dataset.

Each model uses a scikit-learn pipeline combining:

- imputation (missing values)
- scaling (numeric features)
- one-hot encoding (categorical features)
- logistic regression, random forest, or gradient boosting classifier
- hyperparameter tuning (CV search on ROC-AUC)
- threshold tuning on a validation split (optimize F1 instead of fixed 0.5)

---

## Dataset

This project uses the **Heart Disease Dataset** available on Kaggle:

https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

The dataset aggregates several heart disease cohorts (Cleveland, Hungary, Switzerland, and others) originally published in the UCI Machine Learning Repository:

Dua, D. and Graff, C. (2019). *UCI Machine Learning Repository*. University of California, Irvine, School of Information and Computer Sciences.  
https://archive.ics.uci.edu/ml/datasets/Heart+Disease

The dataset is downloaded programmatically using `kagglehub` and is not stored in this repository.

## Results

After removing dataset-origin features to avoid cohort bias, both models perform strongly:

- **Logistic Regression**
  - Test ROC-AUC: 0.899
  - Accuracy (tuned threshold): 0.777
  - 5-fold CV ROC-AUC (best search): 0.888
  - Tuned threshold: 0.29
- **Random Forest**
  - Test ROC-AUC: 0.918
  - Accuracy (tuned threshold): 0.799
  - 5-fold CV ROC-AUC (best search): 0.887
  - Tuned threshold: 0.38

The random forest currently gives the best ROC-AUC on the held-out test set.

### Model Comparison Table

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9176 | 0.7989 | 0.7642 | 0.9216 | 0.8356 | 0.38 |
| Logistic Regression | 0.8985 | 0.7772 | 0.7402 | 0.9216 | 0.8210 | 0.29 |

### ROC Curve
![ROC](reports/figures/roc_curve.png)

### Confusion Matrix
![Confusion Matrix](reports/figures/confusion_matrix.png)

### Feature Importance (Logistic Regression Coefficients)
![Feature Importance](reports/figures/feature_importance.png)

Key predictors include:

- asymptomatic chest pain
- number of blocked vessels (`ca`)
- ST depression during exercise (`oldpeak`)
- exercise-induced angina
- abnormal perfusion test results
- male sex

These align with established cardiology risk factors.

---

## Reproducibility

The project is fully reproducible using a fixed random seed, stratified splits, and a scikit-learn preprocessing pipeline.

### Quick start

```bash
make setup
make download
make train       # trains logistic regression -> models/logreg.joblib
make train_rf    # trains random forest -> models/rf.joblib
make train_gb    # trains gradient boosting -> models/gb.joblib
make eval
make importance  # logistic regression feature importance
make compare     # writes reports/model_comparison.csv
```

### Main outputs

- `models/logreg.joblib`
- `models/rf.joblib`
- `models/gb.joblib`
- `reports/model_comparison.csv`
- `reports/figures/roc_curve.png`
- `reports/figures/confusion_matrix.png`
- `reports/figures/feature_importance.png`
