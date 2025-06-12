import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.copy()
    label_cols = ['gender', 'smoker', 'region']
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col])
    return df
