"""Validated inference using the MLflow-registered breast-cancer model."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import mlflow
import mlflow.sklearn
import pandas as pd

from src.tracking.mlflow_config import configure_mlflow


# These names match the training CSV exactly. Form field names use underscores,
# because HTML field names cannot conveniently contain spaces.
FEATURE_COLUMNS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
    "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
]
FORM_FIELDS = [column.replace(" ", "_") for column in FEATURE_COLUMNS]
MODEL_URI = os.getenv("BREAST_CANCER_MODEL_URI", "models:/BreastCancer_SVM/1")
ARTIFACT_MOUNT_ROOT = os.getenv("BREAST_CANCER_ARTIFACT_MOUNT_ROOT")


class InputValidationError(ValueError):
    """Raised when a request does not contain one valid value per feature."""


@dataclass(frozen=True)
class PredictionResult:
    label: str
    probability: float
    model_uri: str


@lru_cache(maxsize=1)
def get_model():
    """Load the pinned, registered SVM model once per application process."""
    configure_mlflow()
    return mlflow.sklearn.load_model(resolve_model_uri())


def resolve_model_uri() -> str:
    """Resolve a registry model URI, including Windows-to-Docker artifact paths.

    The local MLflow database was created on Windows, so it stores artifact
    locations like ``file:///C:/.../mlruns/...``. Docker receives the same
    artifacts at ``/app/mlruns``. The environment variable is only set in the
    container; local Windows runs retain the original registry URI.
    """
    if not ARTIFACT_MOUNT_ROOT or not MODEL_URI.startswith("models:/"):
        return MODEL_URI

    model_name, version = MODEL_URI.removeprefix("models:/").rsplit("/", maxsplit=1)
    client = mlflow.MlflowClient()
    model_version = client.get_model_version(model_name, version)
    source = getattr(model_version, "storage_location", None)
    if not isinstance(source, str) or not source:
        source = model_version.source
    mount_root = ARTIFACT_MOUNT_ROOT.rstrip("/\\").replace("\\", "/")
    if source.startswith("models:/"):
        model_id = source.removeprefix("models:/").strip("/")
        experiment_id = client.get_run(model_version.run_id).info.experiment_id
        return f"file://{mount_root}/{experiment_id}/models/{model_id}/artifacts"
    normalized_source = source.replace("\\", "/")
    marker = "/mlruns/"
    if marker not in normalized_source:
        return MODEL_URI

    relative_artifact_path = normalized_source.split(marker, maxsplit=1)[1]
    return f"file://{mount_root}/{relative_artifact_path}"


def prepare_features(values: Mapping[str, str]) -> pd.DataFrame:
    """Validate form input and return one feature row in training-column order."""
    missing = [field for field in FORM_FIELDS if field not in values or not values[field].strip()]
    if missing:
        raise InputValidationError(f"Missing required feature values: {', '.join(missing)}")

    unexpected = sorted(set(values) - set(FORM_FIELDS))
    if unexpected:
        raise InputValidationError(f"Unexpected feature values: {', '.join(unexpected)}")

    converted = {}
    for column, field in zip(FEATURE_COLUMNS, FORM_FIELDS):
        try:
            value = float(values[field])
        except (TypeError, ValueError) as error:
            raise InputValidationError(f"{field} must be a number.") from error
        if not math.isfinite(value):
            raise InputValidationError(f"{field} must be a finite number.")
        converted[column] = value

    return pd.DataFrame([converted], columns=FEATURE_COLUMNS)


def predict(values: Mapping[str, str]) -> PredictionResult:
    """Return the clinical class and model confidence for one form submission."""
    features = prepare_features(values)
    model = get_model()
    predicted_class = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    malignant_probability = float(probabilities[1])

    return PredictionResult(
        label="Malignant" if predicted_class == 1 else "Benign",
        probability=malignant_probability,
        model_uri=MODEL_URI,
    )
