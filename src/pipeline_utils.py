"""Shared scikit-learn preprocessing and model builders.

Each builder returns a full pipeline (impute -> scale -> one-hot -> classifier)
and infers numeric/categorical columns from the supplied frame, so the same
builder works on the full feature set, a single-site subset, or a feature
subset. Data loading lives in :mod:`data`.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 100


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    random_seed: int = RANDOM_SEED,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )


def build_preprocessor(X: pd.DataFrame, sparse_output: bool = True) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )


def build_logreg_pipeline(X: pd.DataFrame, random_seed: int = RANDOM_SEED) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", build_preprocessor(X)),
            ("clf", LogisticRegression(max_iter=3000, random_state=random_seed)),
        ]
    )


def build_rf_pipeline(X: pd.DataFrame, random_seed: int = RANDOM_SEED) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", build_preprocessor(X)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=random_seed,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_hgb_pipeline(X: pd.DataFrame, random_seed: int = RANDOM_SEED) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", build_preprocessor(X, sparse_output=False)),
            ("clf", HistGradientBoostingClassifier(random_state=random_seed)),
        ]
    )
