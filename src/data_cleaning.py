"""Script to clean the dataset"""
import pandas as pd


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataset by removing missing values and duplicates

    Args:
        data (pd.DataFrame): The dataset to be cleaned

    Returns:
        pd.DataFrame: The cleaned dataset
    """
    data = data.dropna()
    data = data.drop_duplicates()
    return data


def main():
    """Main function to clean the dataset"""
    print("Cleaning the training dataset")
    data = pd.read_parquet("data/interim/heart_train.parquet")
    cleaned_data = clean_data(data)
    cleaned_data.to_parquet("data/processed/heart_train_cleaned.parquet",
                            index=False)


if __name__ == "__main__":
    main()
