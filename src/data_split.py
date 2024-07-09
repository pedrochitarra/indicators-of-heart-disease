"""Script to split the Indicators of Heart Disease dataset into training and
testing sets."""
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    """Load the Indicators of Heart Disease dataset"""
    data = pd.read_csv('data/raw/heart_2022_with_nans.csv')
    return data


def split_data(data):
    """Split the data into training and testing sets"""
    df_train, df_test = train_test_split(data, test_size=0.2,
                                         random_state=2506)
    return df_train, df_test


def main():
    """Main function to split the data into training and testing sets"""
    print("Splitting the data into training and testing sets")
    data = load_data()
    df_train, df_test = split_data(data)
    print(f"df_train shape: {df_train.shape}")
    print(f"df_test shape: {df_test.shape}")
    df_train.to_parquet('data/interim/heart_train.parquet', index=False)
    df_test.to_parquet('data/interim/heart_test.parquet', index=False)


if __name__ == '__main__':
    main()
