import pandas as pd
from pathlib import Path


REQUIRED_COLUMNS = [
    "id",
    "diagnosis",
]


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the breast cancer dataset.

    Checks:
    - Dataset is not empty
    - Required columns exist
    - Missing values
    - Duplicate rows
    - Target values
    - Data types
    """

    print("\n========== DATA VALIDATION ==========")

    # 1. Check empty dataset
    if df.empty:
        raise ValueError("Dataset is empty.")

    print("PASS: Dataset is not empty")

    # 2. Check required columns
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("PASS: Required columns are present")

    # 3. Check missing values
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()

    if total_missing > 0:
        print("\nWARNING: Missing values found:")
        print(missing_values[missing_values > 0])
    else:
        print("PASS: No missing values")

    # 4. Check duplicate rows
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        print(f"WARNING: Duplicate rows found: {duplicate_count}")
    else:
        print("PASS: No duplicate rows")

    # 5. Check target values
    valid_targets = {"M", "B"}
    actual_targets = set(df["diagnosis"].unique())

    invalid_targets = actual_targets - valid_targets

    if invalid_targets:
        raise ValueError(
            f"Invalid target values found: {invalid_targets}"
        )

    print("PASS: Target values are valid")

    # 6. Check target distribution
    print("\nTarget distribution:")
    print(df["diagnosis"].value_counts())

    # 7. Check data types
    print("\nData types:")
    print(df.dtypes)

    print("\n========== VALIDATION COMPLETE ==========")

    return True


if __name__ == "__main__":

    data_path = Path("data/raw/breast_cancer.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    validate_data(df)
