import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from monitoring.model_monitor import build_drift_report, log_prediction
from src.prediction.predict import FEATURE_COLUMNS, FORM_FIELDS, PredictionResult


def valid_form_data() -> dict[str, str]:
    return {field: "1.0" for field in FORM_FIELDS}


class TestMonitoring(unittest.TestCase):
    def test_logs_prediction_event(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "events.jsonl"
            with patch("monitoring.model_monitor.MONITORING_DIR", log_path.parent), patch(
                "monitoring.model_monitor.PREDICTION_LOG_PATH", log_path
            ):
                log_prediction(
                    valid_form_data(),
                    PredictionResult("Benign", 0.1, "models:/BreastCancer_SVM/1"),
                )

            event = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(event["prediction"], "Benign")
            self.assertEqual(len(event["features"]), 30)

    def test_detects_large_feature_shift(self):
        reference = pd.DataFrame([{feature: 0.0 for feature in FEATURE_COLUMNS}] * 3)
        production = pd.DataFrame([{feature: 0.0 for feature in FEATURE_COLUMNS}] * 3)
        production["area_mean"] = 10.0

        report = build_drift_report(
            production,
            reference,
            threshold=0.5,
            minimum_observations=3,
        )

        self.assertTrue(report["drift_detected"])
        self.assertEqual(report["drifted_features"][0]["feature"], "area_mean")
