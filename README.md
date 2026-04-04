# Breast Cancer Prediction Web App

## Overview

This project is a machine learning based web application that predicts whether a breast cancer case is Positive or Negative using 30 medical features. The model is trained using a Support Vector Machine (SVM) algorithm and achieves approximately 97% accuracy.

The project includes:
* Data preprocessing and scaling
* Model training using SVM
* Model evaluation using accuracy score and confusion matrix
* Saving the trained model using Pickle
* A Flask web application for real-time prediction
---

## Project Structure

```text
Breast-Cancer-Prediction/
│
├── app.py                  # Flask application
├── model.pkl               # Trained machine learning model
├── requirements.txt        # Required Python libraries
├── .gitignore              # Ignored files and folders
├── Breast_Cancer.ipynb     # Jupyter Notebook for training and evaluation
│
└── templates/
    └── index.html          # Frontend HTML form
```

---
## Machine Learning Workflow

The Jupyter Notebook contains the following steps:

1. Load the breast cancer dataset
2. Perform preprocessing
3. Scale the features
4. Train a Support Vector Machine (SVM) model
5. Evaluate the model
6. Achieve approximately 97% accuracy
7. Generate a confusion matrix
8. Save the trained model as `model.pkl`

---

## Features Used
The model uses the following 30 input features:

* Radius Mean
* Texture Mean
* Perimeter Mean
* Area Mean
* Smoothness Mean
* Compactness Mean
* Concavity Mean
* Concave Points Mean
* Symmetry Mean
* Fractal Dimension Mean
* Radius SE
* Texture SE
* Perimeter SE
* Area SE
* Smoothness SE
* Compactness SE
* Concavity SE
* Concave Points SE
* Symmetry SE
* Fractal Dimension SE
* Radius Worst
* Texture Worst
* Perimeter Worst
* Area Worst
* Smoothness Worst
* Compactness Worst
* Concavity Worst
* Concave Points Worst
* Symmetry Worst
* Fractal Dimension Worst

---

## Technologies Used

* Python 
* Flask 
* NumPy
* Pandas
* Scikit-learn
* Pickle
* HTML

---

## Installation
Clone the repository:

```bash
git clone https://github.com/your-username/Breast-Cancer-Prediction.git
cd Breast-Cancer-Prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Run the Flask application:

```bash
python app.py
```

After running the command, open your browser and go to:

```text
http://127.0.0.1:5000/
```
---

## How the Web App Works

1. The user enters all 30 feature values in the form.
2. The form sends the values to the `/predict` route.
3. The Flask app loads the trained `model.pkl` file.
4. The model predicts the result.
5. The prediction is displayed on the webpage.

Possible outputs:

* `Prediction: Positive`
* `Prediction: Negative`
---

## app.py Explanation

* Loads the trained Pickle model
* Renders the home page
* Accepts user input from the HTML form
* Converts all inputs to floating-point numbers
* Checks that exactly 30 features are provided
* Sends the features to the trained model
* Displays the prediction result

---

## requirements.txt

The following main libraries are used:

```text
Flask==3.1.0
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.2.2
gunicorn==23.0.0
```
---

## .gitignore Example
```text
myenv
```
