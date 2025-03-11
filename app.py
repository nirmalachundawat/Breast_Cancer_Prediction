from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received Form Data:", request.form)  # Debugging print

        # Extract all 30 input features from the form
        float_features = [float(x) for x in request.form.values()]
        if len(float_features) != 30:
            raise ValueError("Expected 30 features, but received a different number.")
        
        final_features = [np.array(float_features)]

        print("Input Features:", final_features)  # Debugging print

        # Make prediction
        prediction = model.predict(final_features)

        print("Model Output:", prediction)  # Debugging print

        # Output result based on the prediction
        output = 'Positive' if prediction[0] == 1 else 'Negative'

        # Render result in the template
        return render_template('index.html', prediction_text=f'Prediction: {output}')

    except Exception as e:
        print("Error:", str(e))  # Debugging error
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
