import pandas as pd
import joblib
import argparse
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import json

def train_model(input_path: str, model_path: str, metrics_path: str):
    df = pd.read_csv(input_path)

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Evaluate on validation set
    val_pred = model.predict_proba(X_val)[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_val, val_pred)
    metrics = {"roc_auc": auc}

    # Save metrics
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    print(f"Validation ROC AUC: {auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train insurance risk model")
    parser.add_argument("--input", type=str, required=True, help="Input cleaned CSV")
    parser.add_argument("--model", type=str, required=True, help="Output model file path")
    parser.add_argument("--metrics", type=str, required=True, help="Output metrics JSON path")
    args = parser.parse_args()

    train_model(args.input, args.model, args.metrics)
