from flask import Flask, request, render_template
import pickle
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
model_path = "model.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)

app = Flask(__name__, template_folder="templates")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        logger.info("Prediction request received")

        # Extract all 30 input features
        float_features = [float(x) for x in request.form.values()]

        if len(float_features) != 30:
            raise ValueError(
                f"Expected 30 features but received {len(float_features)}"
            )

        final_features = np.array([float_features])

        # Make prediction
        prediction = model.predict(final_features)

        logger.info(f"Prediction result: {prediction[0]}")

        output = "Positive" if prediction[0] == 1 else "Negative"

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {output}"
        )

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")

        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=False
    )