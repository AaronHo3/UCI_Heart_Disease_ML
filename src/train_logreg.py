import os
import pandas as pd
from joblib import dump
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

RANDOM_SEED = 100
DATA_PATH = "data/raw/heart_disease_uci.csv"
MODEL_PATH = "models/logreg.joblib"

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}.")
    
    df = pd.read_csv(DATA_PATH)

    # Convert num label to binary
    if "num" not in df.columns:
        raise ValueError("Expected 'num' column not found as label")
    y = (df['num'] > 0).astype(int) # Binary classification: 0 = no disease, 1 = disease

    # Features
    drop_cols = ["num", "dataset"] # Drop original label column
    if "id" in df.columns:
        drop_cols.append("id") # Drop id column if exists"

    X = df.drop(columns=drop_cols)

    # Identify numeric and categorical columns
    cat_cols = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Preprocessing pipelines
    # Median imputation for robustness to outliers, standard scaling for numeric features
    numeric_pipe = Pipeline(steps =[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
      ],
      remainder="drop"
    )

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Cross validation on training set to check for overfitting
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1, return_train_score=False)

    auc_scores = cv_results["test_score"]
    print(f"5-fold CV ROC-AUC: {auc_scores.mean():.4f} +/- {auc_scores.std():.4f}")
    print("Fold AUCs:", [f"{s:.4f}" for s in auc_scores])

    # Fit final model on full training set and evaluate on test set
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, preds)

    os.makedirs("models", exist_ok=True)
    dump(model, MODEL_PATH)

    print("Categorical columns:", cat_cols)
    print("Numeric columns:", num_cols)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Test ROC-AUC : {auc:.3f}")
    print(f"Test Accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
