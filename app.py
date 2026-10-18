import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

with open('model/loan_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() if request.is_json else request.form
    try:
        income = float(data.get('income', 50000))
        credit_score = float(data.get('credit_score', 700))
        loan_amount = float(data.get('loan_amount', 15000))
        dti = float(data.get('dti', 0.25))
        emp_length = float(data.get('emp_length', 5))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid numeric parameters provided.'}), 400

    # Scale and predict
    features = np.array([[income, credit_score, loan_amount, dti, emp_length]])
    features_scaled = scaler.transform(features)
    
    prob_default = float(model.predict_proba(features_scaled)[0][1])
    default_pct = round(prob_default * 100, 1)

    # Risk Tier & Decision Logic
    if default_pct < 25.0:
        verdict = "APPROVED (Prime Rate)"
        verdict_class = "success"
        apr = "8.49% APR"
        risk_tier = "Low Default Risk"
    elif default_pct < 55.0:
        verdict = "APPROVED (Subprime / Collateral Required)"
        verdict_class = "warning"
        apr = "14.25% APR"
        risk_tier = "Moderate Default Risk"
    else:
        verdict = "REJECTED (High Default Probability)"
        verdict_class = "danger"
        apr = "N/A"
        risk_tier = "High Default Risk"

    # Drivers
    drivers = []
    if credit_score < 620:
        drivers.append("Credit score below prime threshold (< 620).")
    if dti > 0.40:
        drivers.append(f"Elevated Debt-to-Income ratio ({round(dti*100, 1)}% > 40%).")
    if (loan_amount / max(income, 1)) > 0.45:
        drivers.append("High Loan-to-Income burden.")
    if not drivers:
        drivers.append("Strong credit history and manageable debt obligations.")

    return jsonify({
        'default_probability': default_pct,
        'verdict': verdict,
        'verdict_class': verdict_class,
        'apr': apr,
        'risk_tier': risk_tier,
        'drivers': drivers
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
