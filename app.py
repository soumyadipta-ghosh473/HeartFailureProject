import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("heart_failure_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Failure Prediction System")

st.write("Enter Patient Details")

# Inputs (must match dataset order)

Age = st.number_input("Age", 1, 120)
Sex = st.number_input("Sex (0 = Female, 1 = Male)", 0, 1)
ChestPainType = st.number_input("Chest Pain Type (encoded value)")
RestingBP = st.number_input("Resting Blood Pressure")
Cholesterol = st.number_input("Cholesterol")
FastingBS = st.number_input("Fasting Blood Sugar (0 or 1)", 0, 1)
RestingECG = st.number_input("Resting ECG (encoded value)")
MaxHR = st.number_input("Max Heart Rate")
ExerciseAngina = st.number_input("Exercise Angina (0 or 1)", 0, 1)
Oldpeak = st.number_input("Oldpeak")
ST_Slope = st.number_input("ST Slope (encoded value)")

if st.button("Predict"):

    features = np.array([[Age, Sex, ChestPainType, RestingBP,
                          Cholesterol, FastingBS, RestingECG,
                          MaxHR, ExerciseAngina, Oldpeak, ST_Slope]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        st.error("High Risk of Heart Failure")
    else:
        st.success("Low Risk of Heart Failure")
