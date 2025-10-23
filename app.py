import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load("model.pkl")

st.title("Customer Renewal Prediction")

# Simple input fields for each feature
perc_cash_credit = st.number_input("Percentage of Premium Paid by Cash/Credit", value=0.5)
age_in_days = st.number_input("Age in Days", value=10000)
premium = st.number_input("Premium Amount", value=5000.0)
late_payment_severity = st.number_input("Late Payment Severity", value=1)
late_payment_ratio = st.number_input("Late Payment Ratio", value=0.2)

# Predict button
if st.button("Predict Renewal"):
    input_data = np.array([[perc_cash_credit, age_in_days, premium, late_payment_severity, late_payment_ratio]])
    prediction = model.predict(input_data)
    result = "Will Not Renew" if prediction[0] == 0 else "Will Renew"
    st.success(f"Prediction: {result}")