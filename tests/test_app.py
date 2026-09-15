import unittest
from unittest.mock import patch

import app as flask_app_module
from src.prediction.predict import FORM_FIELDS, PredictionResult


def valid_form_data() -> dict[str, str]:
    return {field: "1.0" for field in FORM_FIELDS}


class TestPredictionRoute(unittest.TestCase):
    def setUp(self):
        flask_app_module.app.config.update(TESTING=True)
        self.client = flask_app_module.app.test_client()

    @patch("app.run_prediction")
    def test_valid_request_returns_prediction(self, mock_predict):
        mock_predict.return_value = PredictionResult(
            label="Benign",
            probability=0.12,
            model_uri="models:/BreastCancer_SVM/1",
        )

        response = self.client.post("/predict", data=valid_form_data())

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction: Benign", response.data)
        self.assertIn(b"12.0%", response.data)
        mock_predict.assert_called_once()

    def test_missing_feature_returns_400(self):
        values = valid_form_data()
        values.pop("radius_mean")

        response = self.client.post("/predict", data=values)

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Missing required feature values: radius_mean", response.data)

    @patch("app.get_model")
    def test_health_check_returns_ok_when_model_is_available(self, mock_get_model):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"status": "ok"})
        mock_get_model.assert_called_once()
