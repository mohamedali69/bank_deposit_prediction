from flask import Flask, request, render_template
import numpy as np
import joblib

model = joblib.load("xgb_model.pkl")

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        age = int(request.form.get("age", 0))
        balance = float(request.form.get("balance", 0.0))
        day = int(request.form.get("day", 1))
        duration = float(request.form.get("duration", 0.0))
        default_bool = int(request.form.get("default_bool", 0))
        housing_bool = int(request.form.get("housing_bool", 0))
        loan_bool = int(request.form.get("loan_bool", 0))
        campaign_cleaned = float(request.form.get("campaign_cleaned", 0.0))
        previous_cleaned = float(request.form.get("previous_cleaned", 0.0))

        job_features = [int(request.form.get(f"job_{job}", 0)) for job in [
            "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student",
            "technician", "unemployed", "unknown"]]

        marital_features = [int(request.form.get(
            f"marital_{status}", 0)) for status in ["married", "single"]]

        education_features = [int(request.form.get(f"education_{level}", 0)) for level in [
            "secondary", "tertiary", "unknown"]]

        contact_features = [int(request.form.get(f"contact_{method}", 0)) for method in [
            "telephone", "unknown"]]

        month_features = [int(request.form.get(f"month_{month}", 0)) for month in [
            "aug", "dec", "feb", "jan", "jul", "jun",
            "mar", "may", "nov", "oct", "sep"]]

        poutcome_features = [int(request.form.get(f"poutcome_{outcome}", 0)) for outcome in [
            "other", "success", "unknown"]]

        features = [
            age, balance, day, duration, default_bool, housing_bool, loan_bool,
            campaign_cleaned, previous_cleaned
        ] + job_features + marital_features + education_features + contact_features + month_features + poutcome_features

        if len(features) != 41:
            return render_template("index.html", prediction="Feature mismatch. Please check input values.")

        features = np.array(features).reshape(1, -1)

        prediction = model.predict_proba(features)[0][1]
        prediction_percentage = prediction * 100

        return render_template("index.html", prediction=f"Probability of deposit: {prediction_percentage:.2f}%")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
