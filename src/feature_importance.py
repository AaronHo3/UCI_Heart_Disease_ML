import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pipeline_utils import load_model_bundle

MODEL_PATH = "models/logreg.joblib"

def main():
    model, _, _ = load_model_bundle(MODEL_PATH)

    # Split pipeline
    preprocessor = model.named_steps["prep"]
    clf = model.named_steps["clf"]

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    # Coefficients from logistic regression
    coefs = clf.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficients": coefs,
        "abs_coefficients": np.abs(coefs)
    }).sort_values(by="abs_coefficients", ascending=False)

    print("\nTop features based on absolute coefficient values:\n")
    print(coef_df.head(15))

    # Plot
    top = coef_df.head(15).iloc[::-1] # Reverse for better plotting

    plt.figure(figsize=(8,6))
    plt.barh(top["feature"], top["coefficients"], color="skyblue")
    plt.title("Top Logistic Regression Coefficients")
    plt.xlabel("Coefficient (effect on log-odds of heart disease)")
    plt.tight_layout()

    plt.savefig("reports/figures/feature_importance.png", dpi=200)
    plt.close()

    print("\nSaved plot -> reports/figures/feature_importance.png")
    
if __name__ == "__main__":
    main()
