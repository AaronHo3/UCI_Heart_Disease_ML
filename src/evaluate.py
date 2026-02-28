import os
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    auc,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

RANDOM_SEED = 100
DATA_PATH = "data/raw/heart_disease_uci.csv"
MODEL_PATH = "models/logreg.joblib"

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}.")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
    
    df = pd.read_csv(DATA_PATH)
    
    if "num" not in df.columns:
        raise ValueError("Expected 'num' column not found as label")
    
    # Binary label: disease vs no disease
    y = (df['num'] > 0).astype(int)
    
    # Features
    X = df.drop(columns = ['num', 'id'], errors='ignore') # Drop label and id if exists

    # Use the same split settings as train_logreg.py so test set is consistent
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Load trained model
    model = load(MODEL_PATH)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save figures
    os.makedirs("reports/figures", exist_ok=True)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.title(f"ROC Curve (AUC = {auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("reports/figures/roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Confusion Matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.title("Confusion Matrix (threshold = 0.5)")
    plt.savefig("reports/figures/confusion_matrix.png", dpi=200, bbox_inches='tight')
    plt.close()

    print("Saved figures to reports/figures/:")
    print("- roc_curve.png")
    print("- confusion_matrix.png")

if __name__ == "__main__":
    main()

            
