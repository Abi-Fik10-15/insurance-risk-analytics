import joblib
import pandas as pd
import argparse
from sklearn.metrics import classification_report

def evaluate_model(model_path: str, test_data_path: str):
    model = joblib.load(model_path)
    df = pd.read_csv(test_data_path)

    X_test = df.drop(columns=['target'])
    y_test = df['target']

    preds = model.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True)
    print(classification_report(y_test, preds))

    # Optionally save the report or metrics
    # Here just printing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate insurance risk model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test-data", type=str, required=True, help="Test data CSV")
    args = parser.parse_args()

    evaluate_model(args.model, args.test_data)
