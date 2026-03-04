from pipeline_utils import (
    RANDOM_SEED,
    build_rf_pipeline,
    evaluate_at_threshold,
    load_dataset,
    save_model_bundle,
    split_train_test,
    split_train_validation,
    tune_model,
    tune_threshold,
)

MODEL_PATH = "models/rf.joblib"

def main():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_train_test(X, y, random_seed=RANDOM_SEED)
    base_model = build_rf_pipeline(X, random_seed=RANDOM_SEED)

    param_grid = {
        "clf__n_estimators": [300, 500, 800, 1200],
        "clf__max_depth": [None, 5, 10, 20, 30],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2", None],
        "clf__class_weight": [None, "balanced", "balanced_subsample"],
    }
    tuned_model, best_params, best_cv_auc = tune_model(
        model=base_model,
        X_train=X_train,
        y_train=y_train,
        param_grid=param_grid,
        random_seed=RANDOM_SEED,
        n_iter=30,
        n_jobs=1,
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
    best_threshold, best_val_f1 = tune_threshold(y_val, val_proba)

    tuned_model.fit(X_train, y_train)
    test_proba = tuned_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_at_threshold(y_test, test_proba, threshold=best_threshold)

    save_model_bundle(
        model_path=MODEL_PATH,
        model=tuned_model,
        threshold=best_threshold,
        metadata={
            "best_cv_auc": best_cv_auc,
            "best_params": best_params,
            "val_f1_at_best_threshold": best_val_f1,
        },
    )

    print(f"Saved tuned model bundle -> {MODEL_PATH}")
    print(f"Best CV ROC-AUC: {best_cv_auc:.4f}")
    print(f"Best params: {best_params}")
    print(f"Tuned threshold on validation split: {best_threshold:.2f} (F1={best_val_f1:.4f})")
    print(f"Test ROC-AUC : {test_metrics['auc']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
