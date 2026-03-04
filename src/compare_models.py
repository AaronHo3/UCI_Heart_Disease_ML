import os

import pandas as pd

from pipeline_utils import (
    RANDOM_SEED,
    build_logreg_pipeline,
    build_rf_pipeline,
    evaluate_at_threshold,
    load_dataset,
    split_train_test,
    split_train_validation,
    tune_model,
    tune_threshold,
)


def evaluate_model_with_tuning(
    name: str,
    base_model,
    param_grid: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    n_iter: int | None = None,
    n_jobs: int = 1,
) -> dict:
    tuned_model, best_params, best_cv_auc = tune_model(
        model=base_model,
        X_train=X_train,
        y_train=y_train,
        param_grid=param_grid,
        random_seed=RANDOM_SEED,
        n_iter=n_iter,
        n_jobs=n_jobs,
        cv=5,
    )

    X_fit, X_val, y_fit, y_val = split_train_validation(
        X_train,
        y_train,
        random_seed=RANDOM_SEED,
        val_size=0.2,
    )
    tuned_model.fit(X_fit, y_fit)
    val_proba = tuned_model.predict_proba(X_val)[:, 1]
    threshold, val_f1 = tune_threshold(y_val, val_proba)

    tuned_model.fit(X_train, y_train)
    test_proba = tuned_model.predict_proba(X_test)[:, 1]
    metrics = evaluate_at_threshold(y_test, test_proba, threshold=threshold)

    return {
        "model": name,
        "auc": metrics["auc"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "threshold": threshold,
        "cv_auc": best_cv_auc,
        "val_f1": val_f1,
        "best_params": str(best_params),
    }


def to_markdown_table(df: pd.DataFrame) -> str:
    df_fmt = df.copy()
    for col in ["auc", "accuracy", "precision", "recall", "f1", "threshold", "cv_auc", "val_f1"]:
        df_fmt[col] = df_fmt[col].map(lambda x: f"{x:.4f}")
    try:
        return df_fmt.to_markdown(index=False)
    except ImportError:
        return df_fmt.to_string(index=False)


def main():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_train_test(X, y, random_seed=RANDOM_SEED)

    logreg_model = build_logreg_pipeline(X, random_seed=RANDOM_SEED)
    rf_model = build_rf_pipeline(X, random_seed=RANDOM_SEED)

    logreg_params = {
        "clf__C": [0.01, 0.1, 1.0, 3.0, 10.0, 30.0],
        "clf__class_weight": [None, "balanced"],
        "clf__solver": ["lbfgs"],
    }
    rf_params = {
        "clf__n_estimators": [300, 500, 800, 1200],
        "clf__max_depth": [None, 5, 10, 20, 30],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2", None],
        "clf__class_weight": [None, "balanced", "balanced_subsample"],
    }

    results = [
        evaluate_model_with_tuning(
            name="Logistic Regression",
            base_model=logreg_model,
            param_grid=logreg_params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_iter=None,
            n_jobs=1,
        ),
        evaluate_model_with_tuning(
            name="Random Forest",
            base_model=rf_model,
            param_grid=rf_params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_iter=30,
            n_jobs=1,
        ),
    ]

    results_df = pd.DataFrame(results).sort_values(by="auc", ascending=False)

    os.makedirs("reports", exist_ok=True)
    results_df.to_csv("reports/model_comparison.csv", index=False)

    print("\nModel Comparison (saved to reports/model_comparison.csv)\n")
    print(results_df)
    print("\nMarkdown Table\n")
    print(to_markdown_table(results_df))


if __name__ == "__main__":
    main()
