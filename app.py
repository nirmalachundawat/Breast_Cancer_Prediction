from flask import Flask, jsonify, request, render_template
import logging

from monitoring.model_monitor import log_prediction
from src.prediction.predict import InputValidationError, get_model, predict as run_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    """Report ready only when the MLflow model can be loaded."""
    try:
        get_model()
    except Exception:
        logger.exception("Health check failed while loading MLflow model")
        return jsonify(status="unavailable"), 503
    return jsonify(status="ok"), 200


@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        logger.info("Prediction request received")

        result = run_prediction(request.form.to_dict(flat=True))
        logger.info("Prediction result: %s", result.label)
        try:
            log_prediction(request.form.to_dict(flat=True), result)
        except Exception:
            # Monitoring must not turn a successful clinical prediction into an error.
            logger.exception("Could not write prediction monitoring event")

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {result.label}",
            confidence=f"{result.probability:.1%}",
        )

    except InputValidationError as error:
        logger.warning("Invalid prediction request: %s", error)
        return render_template(
            "index.html",
            error_message=str(error),
        ), 400

    except Exception:
        logger.exception("Prediction failed")

        return render_template(
            "index.html",
            error_message="The prediction service is temporarily unavailable.",
        ), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=False
    )
