import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def load_and_prepare_data(file_path):
    """Load and prepare the breast cancer dataset."""

    df = pd.read_csv(file_path)

    # Remove ID column
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
    """Create the candidate models."""

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
        ]),

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


def evaluate_model(model, X_test, y_test):
    """Calculate evaluation metrics for one model."""

    # Predictions
    y_pred = model.predict(X_test)

    # Probability predictions
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "Confusion Matrix": cm
    }


if __name__ == "__main__":

    # Dataset path
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

    print("\n========== MODEL EVALUATION ==========")

    results = {}

    # Train + evaluate every model
    for name, model in models.items():

        print(f"\nEvaluating {name}...")

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        results[name] = metrics

        print(f"Accuracy : {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall   : {metrics['Recall']:.4f}")
        print(f"F1 Score : {metrics['F1']:.4f}")
        print(f"ROC-AUC  : {metrics['ROC-AUC']:.4f}")

        print("Confusion Matrix:")
        print(metrics["Confusion Matrix"])

    # Create comparison table
    comparison = pd.DataFrame(results).T

    # Remove confusion matrix from table
    comparison = comparison.drop(
        columns=["Confusion Matrix"]
    )

    print("\n========== MODEL COMPARISON ==========")
    print(comparison.round(4))

    # Select best model based on Recall
    best_model = comparison["Recall"].idxmax()

    print("\n========== BEST MODEL ==========")
    print("Best model based on Recall:", best_model)