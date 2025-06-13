import pandas as pd
import argparse

def clean_data(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    # Example cleaning: drop missing target rows, fill na numeric cols with median
    df = df.dropna(subset=['target'])
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean insurance claims data")
    parser.add_argument("--input", type=str, required=True, help="Input raw data CSV")
    parser.add_argument("--output", type=str, required=True, help="Output cleaned CSV")
    args = parser.parse_args()

    clean_data(args.input, args.output)
