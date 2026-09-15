import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df: pd.DataFrame):
    """
    Prepare the breast cancer dataset for machine learning.

    Returns:
        X_train, X_test, y_train, y_test
    """

    print("\n========== PREPROCESSING ==========")

    # -----------------------------
    # 1. Remove identifier column
    # -----------------------------
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    print("✓ Removed ID column")

    # -----------------------------
    # 2. Separate features and target
    # -----------------------------
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")

    # -----------------------------
    # 3. Encode target
    # M = 1 (Malignant)
    # B = 0 (Benign)
    # -----------------------------
    y = y.map({
        "M": 1,
        "B": 0
    })

    if y.isnull().any():
        raise ValueError("Invalid target values found.")

    print("✓ Target encoded: M=1, B=0")

    # -----------------------------
    # 4. Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Testing samples: {len(X_test)}")

    # -----------------------------
    # 5. Feature scaling
    # -----------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("✓ Features standardized")

    print("\n========== PREPROCESSING COMPLETE ==========")

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":

    data_path = Path("data/raw/breast_cancer.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    print("\nFinal shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)