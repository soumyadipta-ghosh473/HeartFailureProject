import streamlit as st
import joblib
import numpy as np
import os
from groq import Groq
import datetime
import csv
import pandas as pd

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Heart Failure AI System",
    page_icon="❤️",
    layout="wide"
)

# ==========================================
# PROFESSIONAL MEDICAL THEME
# ==========================================
st.markdown("""
<style>

/* Remove top padding */
.block-container {
    padding-top: 2rem;
}

/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #1d4350, #2c5364);
    color: #f5f5f5;
}

/* Glass Card */
.block-container {
    background: rgba(255,255,255,0.05);
    padding: 2rem;
    border-radius: 20px;
    backdrop-filter: blur(12px);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3c43, #2c7744);
}

/* Sidebar text white */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Metric styling */
[data-testid="stMetricValue"] {
    font-size: 28px;
    color: #00ffcc;
}

/* Remove default horizontal lines */
hr {
    border: none;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# PROMPT VERSION
# ==========================================
PROMPT_VERSION = "v4.0"

# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load("heart_failure_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================================
# GROQ INITIALIZATION
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Groq API key not configured in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ==========================================
# HEADER
# ==========================================
st.markdown("""
# ❤️ AI-Powered Heart Failure Risk Assessment System
### Hybrid ML + LLM Clinical Intelligence Platform
""")

st.markdown("""
#### 🏥 Empowering Preventive Cardiology Through Artificial Intelligence

This platform integrates **Machine Learning risk prediction**
with **Large Language Model clinical reasoning**
to support early cardiovascular risk identification.

Designed for educational and AI-healthcare research purposes.
""")

st.caption(f"Prompt Version: {PROMPT_VERSION}")
st.markdown("### 📋 Patient Clinical Parameters")

# ==========================================
# INPUT LAYOUT
# ==========================================
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 1, 120)
    Sex = st.number_input("Sex (0 = Female, 1 = Male)", 0, 1)
    ChestPainType = st.number_input("Chest Pain Type", 0, 3)
    RestingBP = st.number_input("Resting Blood Pressure")
    Cholesterol = st.number_input("Cholesterol")

with col2:
    FastingBS = st.number_input("Fasting Blood Sugar", 0, 1)
    RestingECG = st.number_input("Resting ECG", 0, 2)
    MaxHR = st.number_input("Max Heart Rate")
    ExerciseAngina = st.number_input("Exercise Angina", 0, 1)
    Oldpeak = st.number_input("Oldpeak")
    ST_Slope = st.number_input("ST Slope", 0, 2)

# ==========================================
# PREDICTION
# ==========================================
if st.button("🔍 Predict Risk"):

    features = np.array([[Age, Sex, ChestPainType, RestingBP,
                          Cholesterol, FastingBS, RestingECG,
                          MaxHR, ExerciseAngina, Oldpeak, ST_Slope]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]
    confidence_percentage = probability * 100

    risk_text = "High Risk of Heart Failure" if prediction[0] == 1 else "Low Risk of Heart Failure"

    colA, colB = st.columns(2)

    with colA:
        st.metric("Prediction", risk_text)

    with colB:
        st.metric("Confidence", f"{confidence_percentage:.2f}%")

    # ======================================
    # PROFESSIONAL RISK VISUALIZATION
    # ======================================
    st.markdown("### 📊 Risk Probability Assessment")

    if confidence_percentage < 40:
        bar_color = "#2ecc71"   # Green
    elif confidence_percentage < 70:
        bar_color = "#f1c40f"   # Yellow
    else:
        bar_color = "#e74c3c"   # Red

    st.markdown(f"""
    <div style="background-color:#2b3e50;
                border-radius:20px;
                padding:8px;">
        <div style="background-color:{bar_color};
                    width:{confidence_percentage}%;
                    padding:12px;
                    border-radius:15px;
                    text-align:center;
                    font-weight:bold;
                    color:white;">
            {confidence_percentage:.2f}% Risk Probability
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ======================================
    # LOGGING
    # ======================================
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
                "Confidence"
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

    # ======================================
    # DOWNLOAD LOG BUTTON
    # ======================================
    if os.path.exists(log_file):
        with open(log_file, "rb") as f:
            st.download_button(
                label="⬇ Download Prediction Logs",
                data=f,
                file_name="prediction_logs.csv",
                mime="text/csv"
            )

    # ======================================
    # LLM EXPLANATION
    # ======================================
    prompt = f"""
    Prompt Version: {PROMPT_VERSION}

    Patient details:
    Age: {Age}
    Cholesterol: {Cholesterol}
    Blood Pressure: {RestingBP}
    Maximum Heart Rate: {MaxHR}
    Exercise Angina: {ExerciseAngina}

    Prediction: {risk_text}
    Confidence: {confidence_percentage:.2f}%

    Explain in simple medical language.
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )

        explanation = response.choices[0].message.content

        st.markdown("## 🧠 AI Clinical Risk Interpretation")
        st.write(explanation)

    except Exception as e:
        st.error(f"LLM Error: {str(e)}")

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🏥 Clinical AI Assistant")

st.sidebar.markdown("""
Educational medical assistant powered by LLMs.

⚠ Not a replacement for professional medical care.
""")

st.sidebar.success("LLMOps Enabled Deployment")

st.sidebar.markdown(
    "<div style='background: rgba(255,255,255,0.2); padding:10px; border-radius:12px;'>"
    "<b>Developed by Soumyadipta Ghosh</b>"
    "</div>",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.subheader("💬 Ask Medical Questions")

user_question = st.sidebar.text_input("Enter your question:")

if st.sidebar.button("Ask AI"):

    if user_question.strip():
        chat_prompt = f"""
        You are a medical assistant AI.
        Provide educational information only.
        No diagnosis or treatment advice.

        Question: {user_question}
        """

        try:
            chat_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": chat_prompt}],
            )

            st.sidebar.write(chat_response.choices[0].message.content)

        except Exception as e:
            st.sidebar.error(f"Chatbot Error: {str(e)}")
    else:
        st.sidebar.warning("Please enter a question.")

# ==========================================
# DISCLAIMER
# ==========================================
st.markdown("---")
st.info("⚠ This system is for educational purposes only and not a substitute for professional medical diagnosis.")