#Model Deployment - Flask API
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Define paths
models_path = "../models/"
model_filename = "Random_Forest.pkl"  # Change to the best model if needed

# Load the trained model
model_path = os.path.join(models_path, model_filename)
model = joblib.load(model_path)

# Define the Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get transaction data from request
        data = request.get_json()
        df = pd.DataFrame([data])

        # Ensure correct feature order
        feature_file = os.path.join(models_path, "trained_features.txt")
        with open(feature_file, "r") as f:
            trained_features = [line.strip() for line in f.readlines()]
        df = df.reindex(columns=trained_features, fill_value=0) 

        # Make fraud prediction
        prediction = model.predict(df)[0]
        confidence = model.predict_proba(df)[0][prediction]

        # Return prediction result
        return jsonify({"fraud_prediction": int(prediction), "confidence": round(confidence, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

# 📌 Run Flask API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
