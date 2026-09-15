"""One source of truth for local MLflow tracking settings."""

from pathlib import Path

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "mlflow.db"
ARTIFACT_ROOT = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "Breast_Cancer_Classification"

# SQLAlchemy needs a forward-slash path, including on Windows.
TRACKING_URI = f"sqlite:///{DATABASE_PATH.as_posix()}"
ARTIFACT_URI = ARTIFACT_ROOT.resolve().as_uri()


def configure_mlflow() -> None:
    """Connect the training process to this project's MLflow database."""
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)


def select_experiment() -> None:
    """Create the project experiment once, then make it active."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=ARTIFACT_URI,
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
