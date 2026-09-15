"""Prediction logging and lightweight input-data drift monitoring."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from src.prediction.predict import FEATURE_COLUMNS, PredictionResult, prepare_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "breast_cancer.csv"
MONITORING_DIR = PROJECT_ROOT / "artifacts" / "monitoring"
PREDICTION_LOG_PATH = MONITORING_DIR / "prediction_events.jsonl"
DRIFT_REPORT_PATH = MONITORING_DIR / "drift_report.json"
DEFAULT_DRIFT_THRESHOLD = 0.5
MINIMUM_OBSERVATIONS = 30


def load_reference_features() -> pd.DataFrame:
    """Load the training feature distribution used as the drift baseline."""
    reference = pd.read_csv(REFERENCE_DATA_PATH)
    reference = reference.drop(columns=["id", "diagnosis"], errors="ignore")
    return reference.loc[:, FEATURE_COLUMNS]


def log_prediction(values: Mapping[str, str], result: PredictionResult) -> None:
    """Append a local, structured record for an inference event.

    The project receives no patient identifier, name, or address. Do not add
    such fields to this log in a real deployment without privacy controls.
    """
    features = prepare_features(values).iloc[0].to_dict()
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": result.label,
        "malignancy_probability": result.probability,
        "model_uri": result.model_uri,
        "features": features,
    }
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(event) + "\n")


def load_logged_feature_rows(log_path: Path = PREDICTION_LOG_PATH) -> pd.DataFrame:
    """Read logged feature rows; return an empty, correctly shaped frame if absent."""
    if not log_path.exists():
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    events = []
    with log_path.open(encoding="utf-8") as log_file:
        for line in log_file:
            if line.strip():
                events.append(json.loads(line)["features"])
    return pd.DataFrame(events, columns=FEATURE_COLUMNS)


def build_drift_report(
    production_features: pd.DataFrame,
    reference_features: pd.DataFrame,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
    minimum_observations: int = MINIMUM_OBSERVATIONS,
) -> dict:
    """Compare feature means as standardized shifts from the training baseline."""
    if len(production_features) < minimum_observations:
        return {
            "status": "insufficient_data",
            "message": "More prediction events are required before drift can be assessed.",
            "observations": int(len(production_features)),
            "minimum_observations": minimum_observations,
            "threshold": threshold,
            "drifted_features": [],
        }

    reference_mean = reference_features.mean()
    # Avoid division by zero for a constant reference feature.
    reference_std = reference_features.std().replace(0, 1.0)
    standardized_shifts = ((production_features.mean() - reference_mean).abs() / reference_std)
    drifted = standardized_shifts[standardized_shifts >= threshold].sort_values(ascending=False)

    return {
        "status": "ok",
        "observations": int(len(production_features)),
        "threshold": threshold,
        "drift_detected": bool(not drifted.empty),
        "drifted_features": [
            {"feature": feature, "mean_shift_z": round(float(shift), 4)}
            for feature, shift in drifted.items()
        ],
    }


def create_drift_report(threshold: float = DEFAULT_DRIFT_THRESHOLD) -> dict:
    """Create and save a report from all locally logged prediction events."""
    report = build_drift_report(
        production_features=load_logged_feature_rows(),
        reference_features=load_reference_features(),
        threshold=threshold,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    DRIFT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(create_drift_report(), indent=2))
