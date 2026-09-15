import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import src.prediction.predict as prediction_module
from src.prediction.predict import (
    FEATURE_COLUMNS,
    FORM_FIELDS,
    InputValidationError,
    get_model,
    prepare_features,
)


def valid_form_data() -> dict[str, str]:
    """Return one syntactically valid 30-feature submission."""
    return {field: "1.0" for field in FORM_FIELDS}


class TestFeaturePreparation(unittest.TestCase):
    def test_prepares_all_features_in_training_order(self):
        frame = prepare_features(valid_form_data())

        self.assertEqual(list(frame.columns), FEATURE_COLUMNS)
        self.assertEqual(frame.shape, (1, 30))
        self.assertEqual(frame.iloc[0]["concave points_mean"], 1.0)

    def test_rejects_missing_feature(self):
        values = valid_form_data()
        values.pop("radius_mean")

        with self.assertRaisesRegex(InputValidationError, "radius_mean"):
            prepare_features(values)

    def test_rejects_non_numeric_feature(self):
        values = valid_form_data()
        values["texture_mean"] = "not-a-number"

        with self.assertRaisesRegex(InputValidationError, "texture_mean must be a number"):
            prepare_features(values)

    def test_rejects_non_finite_feature(self):
        values = valid_form_data()
        values["area_mean"] = "nan"

        with self.assertRaisesRegex(InputValidationError, "area_mean must be a finite number"):
            prepare_features(values)


class TestRegisteredModel(unittest.TestCase):
    @unittest.skipUnless(
        Path("mlflow.db").exists(),
        "Run MLflow training before testing the local registered model.",
    )
    def test_registered_model_loads(self):
        model = get_model()
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "predict_proba"))

    @patch("src.prediction.predict.mlflow.MlflowClient")
    def test_resolves_windows_artifact_path_for_docker(self, mock_client):
        mock_client.return_value.get_model_version.return_value = Mock(
            source="file:///C:/Users/Nirmala/Breast_Cancer_Prediction/mlruns/1/models/model-id/artifacts"
        )
        previous_root = prediction_module.ARTIFACT_MOUNT_ROOT
        prediction_module.ARTIFACT_MOUNT_ROOT = "/app/mlruns"
        try:
            uri = prediction_module.resolve_model_uri()
        finally:
            prediction_module.ARTIFACT_MOUNT_ROOT = previous_root

        self.assertEqual(uri, "file:///app/mlruns/1/models/model-id/artifacts")
