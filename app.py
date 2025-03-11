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

        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]

        print("Input Features:", final_features)  # Debugging print

        prediction = model.predict(final_features)  # Make prediction

        print("Model Output:", prediction)  # Debugging print

        output = 'Positive' if prediction[0] == 1 else 'Negative'

        return render_template('index.html', prediction_text=f'Prediction: {output}')
    except Exception as e:
        print("Error:", str(e))  # Print error if any
        return render_template('index.html', prediction_text="Error in prediction")

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         int_features = [int(x) for x in request.form.values()]

#         # Make prediction
#         prediction = model.predict(int_features)[0]  # Extract the scalar value
#         output = 'Positive' if prediction == 1 else 'Negative'

#         return render_template('index.html', prediction_text=f'Prediction: {output}')
    
#     except Exception as e:
#         return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
