import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and preprocessing objects
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Renewal Prediction")

# Input fields
perc_cash_credit = st.number_input("Percentage of Premium Paid by Cash/Credit (0.0–1.0)", min_value=0.0, max_value=1.0, value=0.5)
age_in_years = st.number_input("Age in Years (21–102.0)", min_value=21.0, max_value=102.0, value=30.0)
premium = st.number_input("Premium Amount (1200–60000)", min_value=1200.0, max_value=60000.0, value=5000.0)
late_payment_severity = st.number_input("Late Payment Severity (0.0–41.0)", min_value=0.0, max_value=41.0, value=1.0)
late_payment_ratio = st.number_input("Late Payment Ratio (0.0–5.0)", min_value=0.0, max_value=5.0, value=0.2)

# Predict button
if st.button("Predict"):
    # Convert age to days
    age_in_days = age_in_years * 365

    # Create input DataFrame with exact column names and order
    top_features3 = [
        'perc_premium_paid_by_cash_credit',
        'age_in_days',
        'premium',
        'late_payment_severity',
        'late_payment_ratio'
    ]

    input_df = pd.DataFrame([{
        'perc_premium_paid_by_cash_credit': perc_cash_credit,
        'age_in_days': age_in_days,
        'premium': premium,
        'late_payment_severity': late_payment_severity,
        'late_payment_ratio': late_payment_ratio
    }])[top_features3]  # enforce column order

    # Apply preprocessing
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)
    result = "Customer will likely Not Renew" if prediction[0] == 0 else "Customer will likely Renew"
    if prediction[0] == 0:
        st.markdown("<h3 style='color:red;'>Prediction: Customer will likely Not Renew</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:green;'>Prediction: Customer will likely Renew</h3>", unsafe_allow_html=True)
