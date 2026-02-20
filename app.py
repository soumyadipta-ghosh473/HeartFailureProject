import streamlit as st
import joblib
import numpy as np
import os
from groq import Groq

# Load ML model
model = joblib.load("heart_failure_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize Groq client using environment variable
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.title("Heart Failure Prediction with AI Clinical Explanation")

st.write("Enter Patient Details")

# Inputs
Age = st.number_input("Age", 1, 120)
Sex = st.number_input("Sex (0 = Female, 1 = Male)", 0, 1)
ChestPainType = st.number_input("Chest Pain Type (encoded value)", 0, 3)
RestingBP = st.number_input("Resting Blood Pressure")
Cholesterol = st.number_input("Cholesterol")
FastingBS = st.number_input("Fasting Blood Sugar (0 or 1)", 0, 1)
RestingECG = st.number_input("Resting ECG (encoded value)", 0, 2)
MaxHR = st.number_input("Max Heart Rate")
ExerciseAngina = st.number_input("Exercise Angina (0 or 1)", 0, 1)
Oldpeak = st.number_input("Oldpeak")
ST_Slope = st.number_input("ST Slope (encoded value)", 0, 2)

if st.button("Predict"):

    features = np.array([[Age, Sex, ChestPainType, RestingBP,
                          Cholesterol, FastingBS, RestingECG,
                          MaxHR, ExerciseAngina, Oldpeak, ST_Slope]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        risk_text = "High Risk of Heart Failure"
        st.error(risk_text)
    else:
        risk_text = "Low Risk of Heart Failure"
        st.success(risk_text)

    # Create prompt for LLM explanation
    prompt = f"""
    A patient has the following medical details:
    Age: {Age}
    Cholesterol: {Cholesterol}
    Blood Pressure: {RestingBP}
    Maximum Heart Rate: {MaxHR}
    Exercise Angina: {ExerciseAngina}

    The machine learning model predicted: {risk_text}.

    Explain in simple medical language why this risk level may occur.
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )

        explanation = response.choices[0].message.content

        st.subheader("AI Clinical Explanation")
        st.write(explanation)

    except Exception as e:
        st.error(str(e))
