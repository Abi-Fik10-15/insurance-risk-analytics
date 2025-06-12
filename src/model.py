from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

def split_data(df):
    X = df.drop("charges", axis=1)
    y = df["charges"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "XGBoost": XGBRegressor(objective='reg:squarederror')
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        error = mean_absolute_error(y_test, preds)
        results[name] = error
    return results
