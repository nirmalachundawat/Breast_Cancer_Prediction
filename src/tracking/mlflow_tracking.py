import mlflow
import mlflow.sklearn

from pathlib import Path

from mlflow_config import configure_mlflow, select_experiment

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
)


def load_data(file_path):
    """Load and prepare the breast cancer dataset."""

    import pandas as pd

    df = pd.read_csv(file_path)

    # Remove ID
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    X = df.drop(columns=["diagnosis"])

    y = df["diagnosis"].map({
        "M": 1,
        "B": 0
    })

    if y.isnull().any():
        raise ValueError("Invalid target values found.")

    return X, y


def create_models():

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

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            y_prob
        )
    }

    return metrics


if __name__ == "__main__":
    configure_mlflow()

    # ==========================================
    # 1. DATA
    # ==========================================

    data_path = Path(__file__).resolve().parents[2] / "data/raw/breast_cancer.csv"

    X, y = load_data(data_path)

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


    # ==========================================
    # 2. MLFLOW EXPERIMENT
    # ==========================================

    select_experiment()


    # ==========================================
    # 3. CREATE MODELS
    # ==========================================

    models = create_models()


    # ==========================================
    # 4. TRAIN + TRACK
    # ==========================================

    for name, model in models.items():

        print(f"\nTraining {name}...")

        with mlflow.start_run(
            run_name=name
        ):

            # Train
            model.fit(
                X_train,
                y_train
            )

            # Evaluate
            metrics = evaluate_model(
                model,
                X_test,
                y_test
            )

            # ----------------------------------
            # Log metrics
            # ----------------------------------

            mlflow.log_metrics(metrics)

            # ----------------------------------
            # Log model
            # ----------------------------------
            mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name=f"BreastCancer_{name}",
                skops_trusted_types=[
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier"
                ]
            )

            # ----------------------------------
            # Print results
            # ----------------------------------

            print(
                f"Accuracy : {metrics['accuracy']:.4f}"
            )

            print(
                f"Precision: {metrics['precision']:.4f}"
            )

            print(
                f"Recall   : {metrics['recall']:.4f}"
            )

            print(
                f"F1 Score : {metrics['f1_score']:.4f}"
            )

            print(
                f"ROC-AUC  : {metrics['roc_auc']:.4f}"
            )

            print(
                f"{name} logged to MLflow"
            )


    print("\n================================")
    print("MLFLOW TRACKING COMPLETE")
    print("================================")
