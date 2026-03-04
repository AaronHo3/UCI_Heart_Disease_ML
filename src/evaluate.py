import os
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

MODEL_PATH = "models/logreg.joblib"

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")

    X, y = load_dataset()
    _, X_test, _, y_test = split_train_test(X, y)
    model, threshold, _ = load_model_bundle(MODEL_PATH)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_at_threshold(y_test, proba, threshold=threshold)
    preds = (proba >= threshold).astype(int)

    print(f"Threshold: {threshold:.2f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    # Save figures
    os.makedirs("reports/figures", exist_ok=True)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.4f}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.title(f"ROC Curve (AUC = {metrics['auc']:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("reports/figures/roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.title(f"Confusion Matrix (threshold = {threshold:.2f})")
    plt.savefig("reports/figures/confusion_matrix.png", dpi=200, bbox_inches='tight')
    plt.close()

    print("Saved figures to reports/figures/:")
    print("- roc_curve.png")
    print("- confusion_matrix.png")

if __name__ == "__main__":
    main()

            
