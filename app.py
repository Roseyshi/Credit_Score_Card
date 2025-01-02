from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
import os

app = Flask(__name__)

# Function to load and preprocess the data
def load_and_train_model():
    # Load and process data
    data = pd.read_csv("clean_train.csv")
    X = data.drop('Credit_Score', axis=1)
    y = data['Credit_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # Save the model to a file
    dump(rf, 'credit_classifier.joblib', compress=('gzip', 9))
    return rf

# Load the trained model (or train it if necessary)
def load_model():
    if os.path.exists('credit_classifier.joblib'):
        return load('credit_classifier.joblib')
    else:
        return load_and_train_model()

loaded_model = load_model()

# Define target names
target_names = {0: "Good", 1: "Poor", 2: "Standard"}

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        outstanding_debt = float(request.form['outstanding_debt'])
        credit_mix = int(request.form['credit_mix'])
        credit_history_age = float(request.form['credit_history_age'])
        monthly_balance = float(request.form['monthly_balance'])
        payment_behaviour = float(request.form['payment_behaviour'])
        annual_income = float(request.form['annual_income'])
        delayed_payments = int(request.form['delayed_payments'])

        # Prepare input data for prediction
        input_data = [[
            outstanding_debt, credit_mix, credit_history_age,
            monthly_balance, payment_behaviour, annual_income,
            delayed_payments
        ]]

        # Predict using the loaded model
        prediction = loaded_model.predict(input_data)[0]  # Use loaded_model here

        # Map the prediction to the category name
        target_names = {0: "Good", 1: "Poor", 2: "Standard"}
        prediction_text = target_names.get(prediction, "Unknown")

        # Render result.html with the prediction
        return render_template('result.html', prediction=prediction_text)
    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
