import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def load_and_prepare_data(file_path):
    """Load and prepare the breast cancer dataset."""

    df = pd.read_csv(file_path)

    # Remove identifier
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Separate features and target
    X = df.drop(columns=["diagnosis"])

    y = df["diagnosis"].map({
        "M": 1,
        "B": 0
    })

    if y.isnull().any():
        raise ValueError("Invalid target values found.")

    return X, y


def create_models():
    """Create the candidate ML models."""

    models = {

        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ]),

        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ))
        ]),   # <-- comma is important

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                probability=True,
                random_state=42
            ))
        ]),

        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            ))
        ])
    }

    return models


def train_models(X_train, y_train, models):
    """Train all candidate models."""

    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        trained_models[name] = model

        print(f"{name} trained successfully")

    return trained_models


if __name__ == "__main__":

    data_path = Path("data/raw/breast_cancer.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    # Load data
    X, y = load_and_prepare_data(data_path)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Dataset loaded")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Features:", X.shape[1])

    # Create models
    models = create_models()

    # Train models
    trained_models = train_models(
        X_train,
        y_train,
        models
    )

    print("\n========== TRAINING COMPLETE ==========")

    for name in trained_models:
        print(name)
