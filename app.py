from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("network_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    users = float(request.form["users"])
    download = float(request.form["download"])
    upload = float(request.form["upload"])
    latency = float(request.form["latency"])
    weather = float(request.form["weather"])

    # Prepare input data
    data = np.array([[users, download, upload, latency, weather]])

    # Make prediction
    prediction = model.predict(data)[0]

    # Result and alert
    if prediction == 1:
        result = "Network Failure Detected"
        alert = "🚨 ALERT: Mobile Network Failure Detected!"
    else:
        result = "Network is Normal"
        alert = "✅ Network is Normal"

    return render_template(
        "index.html",
        prediction=result,
        alert=alert
    )


if __name__ == "__main__":
    app.run(debug=True)