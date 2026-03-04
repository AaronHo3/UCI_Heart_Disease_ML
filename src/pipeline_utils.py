import os
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 100
DATA_PATH = "data/raw/heart_disease_uci.csv"
DROP_COLS = ["num", "dataset"]
DEFAULT_THRESHOLD = 0.5


def load_dataset(data_path: str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}.")

    df = pd.read_csv(data_path)
    if "num" not in df.columns:
        raise ValueError("Expected 'num' column not found as label")

    y = (df["num"] > 0).astype(int)
    drop_cols = DROP_COLS + (["id"] if "id" in df.columns else [])
    X = df.drop(columns=drop_cols)
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    random_seed: int = RANDOM_SEED,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )


def split_train_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_seed: int = RANDOM_SEED,
    val_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_seed,
        stratify=y_train,
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
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
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
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


def tune_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict[str, Any],
    random_seed: int = RANDOM_SEED,
    n_iter: int | None = None,
    n_jobs: int = 1,
    cv: int = 5,
) -> tuple[Pipeline, dict[str, Any], float]:
    if n_iter is None:
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
        )
    else:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=random_seed,
            n_jobs=n_jobs,
            refit=True,
        )

    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_threshold(
    y_true: pd.Series,
    proba: np.ndarray,
    min_threshold: float = 0.1,
    max_threshold: float = 0.9,
    step: float = 0.01,
) -> tuple[float, float]:
    thresholds = np.arange(min_threshold, max_threshold + step, step)
    best_threshold = DEFAULT_THRESHOLD
    best_f1 = -1.0

    for threshold in thresholds:
        preds = (proba >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        # Tie-break toward a threshold closest to 0.5 for stability.
        if score > best_f1 or (score == best_f1 and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, float(best_f1)


def evaluate_at_threshold(
    y_true: pd.Series,
    proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    preds = (proba >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, proba),
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
    }


def save_model_bundle(
    model_path: str,
    model: Pipeline,
    threshold: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    payload = {
        "model": model,
        "threshold": float(threshold),
        "metadata": metadata or {},
    }
    dump(payload, model_path)


def load_model_bundle(model_path: str) -> tuple[Pipeline, float, dict[str, Any]]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}.")

    obj = load(model_path)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        threshold = float(obj.get("threshold", DEFAULT_THRESHOLD))
        metadata = obj.get("metadata", {})
        return model, threshold, metadata

    return obj, DEFAULT_THRESHOLD, {}
