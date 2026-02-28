import os
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

RANDOM_SEED = 100
DATA_PATH = "data/raw/heart_disease_uci.csv"
DROP_COLS = ["num", "dataset"] # Drop original label column

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

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
    return preprocessor

def evaluate_model(name:str, model: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
    cv_auc_mean = cv_res["test_score"].mean()
    cv_auc_std = cv_res["test_score"].std()
    
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    return {
        "model": name,
        "auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0)
    }
    
def to_markdown_table(df: pd.DataFrame) -> str:
    df_fmt = df.copy()
    for col in ["auc", "accuracy", "precision", "recall", "f1"]:
        df_fmt[col] = df_fmt[col].map(lambda x: f"{x:.4f}")
    try:
        return df_fmt.to_markdown(index=False)
    except ImportError:
        # Fall back when optional 'tabulate' dependency is unavailable.
        return df_fmt.to_string(index=False)

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}.")
    
    df = pd.read_csv(DATA_PATH)

    if "num" not in df.columns:
        raise ValueError("Expected 'num' column not found as label")
    
    y = (df['num'] > 0).astype(int) # Binary classification: 0 = no disease, 1 = disease
    X = df.drop(columns=DROP_COLS + (["id"] if "id" in df.columns else []))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    preprocessor = build_preprocessor(X)

    logreg_model = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", LogisticRegression(random_state=RANDOM_SEED, max_iter=1000))
    ])

    rf_model = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=500, 
        random_state=RANDOM_SEED,
        class_weight="balanced",
        n_jobs=-1))
    ])

    results = []
    for name, model in [("Logistic Regression", logreg_model), ("Random Forest", rf_model)]:
        res = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        results.append(res)

    results_df = pd.DataFrame(results).sort_values(by="auc", ascending=False)

    os.makedirs("reports", exist_ok=True)
    results_df.to_csv("reports/model_comparison.csv", index=False)

    print("\nModel Comparison (saved to reports/model_comparison.csv)\n")
    print(results_df)

    print("\nMarkdown Table\n")
    print(to_markdown_table(results_df))

if __name__ == "__main__":
    main()
