import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load("model.pkl")

st.title("Customer Renewal Prediction")

# Input fields with default values
perc_cash_credit = st.number_input("Percentage of Premium Paid by Cash/Credit (0.0–1.0)", value=0.5)
age_in_years = st.number_input("Age in Years (21.015625–102.0)", value=30.0)
premium = st.number_input("Premium Amount (1200–60000)", value=5000.0)
late_payment_severity = st.number_input("Late Payment Severity (0.0–41.0)", value=1.0)
late_payment_ratio = st.number_input("Late Payment Ratio (0.0–5.0)", value=0.2)

# Predict button
if st.button("Predict"):
    # Validation
    if not (0.0 <= perc_cash_credit <= 1.0):
        st.error("Percentage of Premium Paid by Cash/Credit must be between 0.0 and 1.0")
    elif not (21.015625 <= age_in_years <= 102.0):
        st.error("Age in Years must be between 21.015625 and 102.0")
    elif not (1200 <= premium <= 60000):
        st.error("Premium must be between 1200 and 60000")
    elif not (0.0 <= late_payment_severity <= 41.0):
        st.error("Late Payment Severity must be between 0.0 and 41.0")
    elif not (0.0 <= late_payment_ratio <= 5.0):
        st.error("Late Payment Ratio must be between 0.0 and 5.0")
    else:
        input_data = np.array([[perc_cash_credit, age_in_years, premium, late_payment_severity, late_payment_ratio]])
        prediction = model.predict(input_data)
        result = "Will Not Renew" if prediction[0] == 0 else "Will Renew"
        st.success(f"Prediction: {result}")