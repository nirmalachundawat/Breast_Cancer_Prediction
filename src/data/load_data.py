import pandas as pd
from pathlib import Path


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the breast cancer dataset from a CSV file.

    Parameters:
        file_path: Path to the CSV dataset.

    Returns:
        pandas DataFrame containing the dataset.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    print(f"Dataset loaded successfully: {path}")
    print(f"Shape: {df.shape}")

    return df


if __name__ == "__main__":
    data_path = "data/raw/breast_cancer.csv"

    df = load_data(data_path)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())