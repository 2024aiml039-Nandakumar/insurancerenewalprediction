import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and preprocessing objects
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.title("Customer Renewal Prediction")

# Input fields
perc_cash_credit = st.number_input("Percentage of Premium Paid by Cash/Credit (0.0–1.0)", value=0.5)
age_in_years = st.number_input("Age in Years (21.015625–102.0)", value=30.0)
premium = st.number_input("Premium Amount (1200–60000)", value=5000.0)
late_payment_severity = st.number_input("Late Payment Severity (0.0–41.0)", value=1.0)
late_payment_ratio = st.number_input("Late Payment Ratio (0.0–5.0)", value=0.2)

# Predict button
if st.button("Predict"):
    # Validate inputs
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
        # Create input DataFrame
        input_dict = {
            'perc_premium_paid_by_cash_credit': [perc_cash_credit],
            'age_in_days': [age_in_years * 365],  # Convert years to days
            'premium': [premium],
            'late_payment_severity': [late_payment_severity],
            'late_payment_ratio': [late_payment_ratio]
        }
        input_df = pd.DataFrame(input_dict)

        # Identify column types
        categorical_cols = input_df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = input_df.select_dtypes(include=['number']).columns.tolist()

        # Apply preprocessing
        input_cat = target_encoder.transform(input_df[categorical_cols]) if categorical_cols else pd.DataFrame()
        input_num = scaler.transform(input_df[numerical_cols]) if numerical_cols else pd.DataFrame()

        # Combine processed features
        input_processed = np.hstack([input_num, input_cat]) if not input_cat.empty else input_num

        # Predict
        prediction = model.predict(input_processed)
        result = "Customer will likely Not Renew" if prediction[0] == 0 else "Customer will likely Renew"
        st.success(f"Prediction: {result}")