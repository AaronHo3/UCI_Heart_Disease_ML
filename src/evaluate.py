import os
import shutil
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    ConfusionMatrixDisplay,
)

from pipeline_utils import (
    evaluate_at_threshold,
    load_dataset,
    load_model_bundle,
    split_train_test,
)

MODEL_CONFIGS = [
    ("Logistic Regression", "models/logreg.joblib", "logreg"),
    ("Random Forest", "models/rf.joblib", "rf"),
    ("Gradient Boosting", "models/gb.joblib", "gb"),
]


def main():
    X, y = load_dataset()
    _, X_test, _, y_test = split_train_test(X, y)
    os.makedirs("reports/figures", exist_ok=True)

    available_models = []
    for model_name, model_path, model_tag in MODEL_CONFIGS:
        if not os.path.exists(model_path):
            print(f"Skipping {model_name}: model file not found at {model_path}.")
            continue

        model, threshold, _ = load_model_bundle(model_path)
        proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_at_threshold(y_test, proba, threshold=threshold)
        preds = (proba >= threshold).astype(int)
        fpr, tpr, _ = roc_curve(y_test, proba)

        available_models.append((model_name, model_tag, threshold, metrics, fpr, tpr, preds))

    if not available_models:
        raise FileNotFoundError("No model bundles found under models/. Train at least one model first.")

    for model_name, model_tag, threshold, metrics, fpr, tpr, preds in available_models:
        print(f"\n{model_name}")
        print(f"Threshold: {threshold:.2f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title(f"ROC Curve - {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.savefig(f"reports/figures/roc_curve_{model_tag}.png", dpi=200, bbox_inches="tight")
        plt.close()

        ConfusionMatrixDisplay.from_predictions(y_test, preds)
        plt.title(f"Confusion Matrix - {model_name} (threshold = {threshold:.2f})")
        plt.savefig(f"reports/figures/confusion_matrix_{model_tag}.png", dpi=200, bbox_inches="tight")
        plt.close()

    # Combined ROC chart for side-by-side comparison
    plt.figure()
    for model_name, _, _, metrics, fpr, tpr, _ in available_models:
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={metrics['auc']:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curves - All Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig("reports/figures/roc_curve_all_models.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Backward-compatible filenames retained for README links.
    for model_name, model_tag, _, _, _, _, _ in available_models:
        if model_tag == "logreg":
            shutil.copyfile("reports/figures/roc_curve_logreg.png", "reports/figures/roc_curve.png")
            shutil.copyfile("reports/figures/confusion_matrix_logreg.png", "reports/figures/confusion_matrix.png")
            print("\nBack-compat outputs:")
            print("- roc_curve.png (Logistic Regression)")
            print("- confusion_matrix.png (Logistic Regression)")
            break

    print("\nSaved figures to reports/figures/:")
    print("- roc_curve_all_models.png")
    print("- roc_curve_logreg.png / roc_curve_rf.png / roc_curve_gb.png")
    print("- confusion_matrix_logreg.png / confusion_matrix_rf.png / confusion_matrix_gb.png")

if __name__ == "__main__":
    main()

            
