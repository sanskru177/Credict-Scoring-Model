import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved models
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Credit Scoring Predictor", layout="centered")
st.title("💳 Credit Scoring Prediction App")
st.write("Enter applicant's financial and demographic details to predict creditworthiness.")

# Input form
with st.form("credit_form"):
    income = st.number_input("Monthly Income (USD)", min_value=0)
    loan_amount = st.number_input("Loan Amount (USD)", min_value=0)
    loan_term = st.number_input("Loan Term (months)", min_value=1)
    credit_history = st.number_input("Credit History Length (years)", min_value=0)
    number_of_defaults = st.number_input("Number of Past Defaults", min_value=0)
    debt_to_income_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.2)
    age = st.number_input("Age", min_value=18, max_value=100)
    
    employment_status = st.selectbox("Employment Status", ["employed", "self-employed", "unemployed"])
    education = st.selectbox("Education Level", ["graduate", "undergraduate", "highschool", "none"])
    
    submitted = st.form_submit_button("Predict")

# Mapping categorical to numeric
employment_dict = {"employed": 0, "self-employed": 1, "unemployed": 2}
education_dict = {"graduate": 0, "undergraduate": 1, "highschool": 2, "none": 3}

if submitted:
    # Create input array
    input_data = pd.DataFrame([[
        income, loan_amount, loan_term, credit_history, number_of_defaults,
        debt_to_income_ratio, age,
        employment_dict[employment_status], education_dict[education]
    ]], columns=[
        'income', 'loan_amount', 'loan_term', 'credit_history', 'number_of_defaults',
        'debt_to_income_ratio', 'age', 'employment_status', 'education'
    ])
    
    # Scale the data
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Output
    if prediction == 1:
        st.success(f"✅ Applicant is likely creditworthy. Confidence: {probability:.2f}")
    else:
        st.error(f"❌ Applicant is at high credit risk. Confidence: {1 - probability:.2f}")
