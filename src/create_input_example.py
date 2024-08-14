"""Get a sample from train inputs and save as a dict to test the API."""
import json

import pandas as pd


def main():
    """Function to get a sample from train inputs and save as a dict."""
    # Sample data
    df_train = pd.read_parquet("data/processed/heart_train_cleaned.parquet")
    sample = df_train.sample(1).to_dict(orient="records")[0]

    # Save sample as a dict
    with open("sample_input.json", "w", encoding="utf-8") as f:
        json.dump(sample, f)


if __name__ == "__main__":
    main()
