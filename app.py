from flask import Flask, request, render_template
import numpy as np
import joblib

model = joblib.load("xgb_model.pkl")

app = Flask(__name__)


@app.route("/")
def index():
    default_form_data = {
        "age": 18,
        "balance": 0.0,
        "day": 1,
        "duration": 0.0,
        "housing_bool": 0,
        "loan_bool": 0,
        "job": "blue-collar",
        "marital": "single",
        "education": "secondary",
        "contact": "cellular",
        "month": "jan",
        "poutcome": "unknown",
        "campaign_cleaned": 0.0,
        "previous_cleaned": 0.0
    }
    return render_template("index.html", form_data=default_form_data)


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        form_data = {
            "age": int(request.form.get("age", 0)),
            "balance": float(request.form.get("balance", 0.0)),
            "day": int(request.form.get("day", 1)),
            "duration": float(request.form.get("duration", 0.0)),
            "housing_bool": int(request.form.get("housing_bool", 0)),
            "loan_bool": int(request.form.get("loan_bool", 0)),
            "job": request.form.get("job", "blue-collar"),
            "marital": request.form.get("marital", "single"),
            "education": request.form.get("education", "secondary"),
            "contact": request.form.get("contact", "cellular"),
            "month": request.form.get("month", "jan"),
            "poutcome": request.form.get("poutcome", "unknown"),
            "campaign_cleaned": float(request.form.get("campaign_cleaned", 0.0)),
            "previous_cleaned": float(request.form.get("previous_cleaned", 0.0)),
        }
        job_features = [0 if request.form.get("job") != job else 1 for job in [
            "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student",
            "technician", "unemployed", "unknown"]]
        marital_features = [0 if request.form.get(
            "marital") != status else 1 for status in ["married", "single"]]
        education_features = [0 if request.form.get(
            "education") != level else 1 for level in [
            "secondary", "tertiary", "unknown"]]
        contact_features = [0 if request.form.get(
            "contact") != method else 1 for method in [
            "telephone", "unknown"]]
        month_features = [0 if request.form.get(
            "month") != month else 1 for month in [
            "aug", "dec", "feb", "jan", "jul", "jun",
            "mar", "may", "nov", "oct", "sep"]]
        poutcome_features = [0 if request.form.get(
            "poutcome") != outcome else 1 for outcome in [
            "other", "success", "unknown"]]

        features = [
            form_data["age"], form_data["balance"], form_data["day"], form_data["duration"], 0,
            form_data["housing_bool"], form_data["loan_bool"], form_data["campaign_cleaned"],
            form_data["previous_cleaned"]
        ] + job_features + marital_features + education_features + contact_features + month_features + poutcome_features

        features = np.array(features, dtype=np.float32).reshape(1, -1)

        prediction = model.predict_proba(features)[0][1]
        prediction_percentage = prediction * 100
        return render_template("result.html", prediction=f"Probability of deposit: {prediction_percentage:.2f}%", form_data=request.form)

    return render_template("index.html", form_data=request.form)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
