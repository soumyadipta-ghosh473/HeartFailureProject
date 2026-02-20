import streamlit as st
import joblib
import numpy as np
import os
from groq import Groq
import datetime
import csv

# =========================
# Prompt Version Control
# =========================
PROMPT_VERSION = "v1.0"

# =========================
# Load ML Model & Scaler
# =========================
model = joblib.load("heart_failure_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# Initialize Groq Client
# =========================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Groq API key not set. Please run: set GROQ_API_KEY=your_key")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# =========================
# UI - Main Section
# =========================
st.title("Heart Failure Prediction with AI Clinical Explanation")
st.write("Enter Patient Details")
st.caption(f"Prompt Version: {PROMPT_VERSION}")

# =========================
# User Inputs
# =========================
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

# =========================
# Prediction Block
# =========================
if st.button("Predict"):

    features = np.array([[Age, Sex, ChestPainType, RestingBP,
                          Cholesterol, FastingBS, RestingECG,
                          MaxHR, ExerciseAngina, Oldpeak, ST_Slope]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]
    confidence_percentage = probability * 100

    if prediction[0] == 1:
        risk_text = "High Risk of Heart Failure"
        st.error(risk_text)
    else:
        risk_text = "Low Risk of Heart Failure"
        st.success(risk_text)

    st.write(f"Model Confidence: {confidence_percentage:.2f}%")

    # =========================
    # Logging (Monitoring)
    # =========================
    log_file = "prediction_logs.csv"
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Prompt_Version",
                "Age",
                "Cholesterol",
                "RestingBP",
                "Prediction",
                "Confidence_Percentage"
            ])

        writer.writerow([
            datetime.datetime.now(),
            PROMPT_VERSION,
            Age,
            Cholesterol,
            RestingBP,
            risk_text,
            round(confidence_percentage, 2)
        ])

    # =========================
    # LLM Explanation Prompt
    # =========================
    prompt = f"""
    Prompt Version: {PROMPT_VERSION}

    A patient has the following medical details:
    Age: {Age}
    Cholesterol: {Cholesterol}
    Blood Pressure: {RestingBP}
    Maximum Heart Rate: {MaxHR}
    Exercise Angina: {ExerciseAngina}

    The machine learning model predicted: {risk_text}
    with a confidence of {confidence_percentage:.2f}%.

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
        st.error(f"LLM Error: {str(e)}")


# =========================
# Sidebar Medical Chatbot
# =========================
with st.sidebar:
    st.header("💬 Medical Assistant Chatbot")

    user_question = st.text_input("Ask a medical question:")

    if st.button("Ask"):

        if user_question.strip() == "":
            st.warning("Please enter a question.")
        else:
            chat_prompt = f"""
            You are a medical assistant AI.

            Provide clear, simple, educational information.
            Do NOT provide medical diagnosis.
            Avoid definitive treatment instructions.

            Question: {user_question}
            """

            try:
                chat_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": chat_prompt}],
                )

                answer = chat_response.choices[0].message.content
                st.write(answer)

            except Exception as e:
                st.error(f"Chatbot Error: {str(e)}")


# =========================
# Disclaimer
# =========================
st.info("⚠ This system is for educational purposes only and not a substitute for professional medical diagnosis.")