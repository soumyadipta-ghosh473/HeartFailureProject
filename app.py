import streamlit as st
import joblib
import numpy as np
import os
from groq import Groq
import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Heart Failure AI Platform",
    page_icon="❤️",
    layout="wide"
)

# ==========================================
# THEME
# ==========================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1d4350, #2c5364);
    color: white;
}
.stButton>button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 10px;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3c43, #2c7744);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# SYSTEM INFO
# ==========================================
PROMPT_VERSION = "v14.0"
MODEL_VERSION = "HeartFailure-XGB-v1"
LLM_MODEL = "llama-3.3-70b-versatile"

st.title("❤️ AI-Powered Heart Failure Risk Assessment Platform")

doctor_mode = st.toggle("Doctor Mode (Technical Explanation)", value=False)

st.markdown(f"""
**ML Model:** {MODEL_VERSION}  
**Prompt Version:** {PROMPT_VERSION}  
**LLM Model:** {LLM_MODEL}  
**Deployment:** Streamlit Cloud  
""")

# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load("heart_failure_model.pkl")
scaler = joblib.load("scaler.pkl")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Groq API key not configured.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ==========================================
# SESSION LOG STORAGE
# ==========================================
if "logs" not in st.session_state:
    st.session_state.logs = []

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["🔍 Risk Assessment", "📊 Analytics Dashboard", "⚙ System Info"])

# ==========================================
# TAB 1 — RISK ASSESSMENT
# ==========================================
with tab1:

    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", 1, 120)
        Sex = st.number_input("Sex (0=Female,1=Male)", 0, 1)
        ChestPainType = st.number_input("Chest Pain Type (0–3)", 0, 3)
        RestingBP = st.number_input("Resting Blood Pressure")
        Cholesterol = st.number_input("Cholesterol")

    with col2:
        FastingBS = st.number_input("Fasting Blood Sugar (0 or 1)", 0, 1)
        RestingECG = st.number_input("Resting ECG (0–2)", 0, 2)
        MaxHR = st.number_input("Max Heart Rate")
        ExerciseAngina = st.number_input("Exercise Angina (0 or 1)", 0, 1)
        Oldpeak = st.number_input("Oldpeak")
        ST_Slope = st.number_input("ST Slope (0–2)", 0, 2)

    st.markdown("### 🔄 What-If Simulation")
    chol_increase = st.slider("Increase Cholesterol (%)", 0, 100, 0)
    adjusted_chol = Cholesterol * (1 + chol_increase / 100)

    if st.button("Predict Risk"):

        features = np.array([[Age, Sex, ChestPainType, RestingBP,
                              adjusted_chol, FastingBS, RestingECG,
                              MaxHR, ExerciseAngina, Oldpeak, ST_Slope]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0][1]
        confidence = probability * 100

        risk_text = "High Risk" if prediction[0] == 1 else "Low Risk"

        st.metric("Prediction", risk_text)
        st.metric("Confidence", f"{confidence:.2f}%")

        # Store log
        st.session_state.logs.append({
            "Timestamp": datetime.datetime.now(),
            "Prediction": risk_text,
            "Confidence": confidence
        })

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text': "Risk Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"},
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # ==========================================
        # DIFFERENT SYSTEM PROMPTS (REAL FIX)
        # ==========================================

        if doctor_mode:
            system_prompt = """
            You are a clinical cardiologist providing technical reasoning.
            Use medical terminology.
            Include pathophysiology explanation.
            Mention risk factors explicitly.
            Structure response with bullet points.
            Avoid simplification.
            """

            user_prompt = f"""
            Patient Data:
            Age: {Age}
            Cholesterol: {adjusted_chol}
            Blood Pressure: {RestingBP}
            Max HR: {MaxHR}
            Oldpeak: {Oldpeak}

            Model Prediction: {risk_text}
            Confidence: {confidence:.2f}%

            Provide a technical cardiology-level explanation.
            """

        else:
            system_prompt = """
            You are a healthcare educator explaining results to a non-medical patient.
            Use simple language.
            Avoid medical jargon.
            Use short sentences.
            Be reassuring but realistic.
            """

            user_prompt = f"""
            The model predicted {risk_text}
            with {confidence:.2f}% confidence.

            Explain in very simple language
            what this means and why it might happen.
            """

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )

        st.markdown("## 🧠 AI Explanation")
        st.write(response.choices[0].message.content)

# ==========================================
# TAB 2 — ANALYTICS
# ==========================================
with tab2:

    st.subheader("📊 Risk Distribution")

    if len(st.session_state.logs) > 0:

        df = pd.DataFrame(st.session_state.logs)

        high = (df["Prediction"] == "High Risk").sum()
        low = (df["Prediction"] == "Low Risk").sum()
        total = len(df)

        st.metric("Total Predictions", total)
        st.metric("High Risk %", round((high / total) * 100, 2))

        fig = px.pie(
            names=["High Risk", "Low Risk"],
            values=[high, low],
            color=["High Risk", "Low Risk"],
            color_discrete_map={"High Risk": "red", "Low Risk": "green"}
        )

        st.plotly_chart(fig, use_container_width=True)

        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇ Download Prediction Logs",
            csv_data,
            "prediction_logs.csv",
            "text/csv"
        )

    else:
        st.info("No predictions yet.")

# ==========================================
# TAB 3 — SYSTEM INFO
# ==========================================
with tab3:
    st.markdown("""
    **System Architecture**
    1. ML Prediction  
    2. LLM Explanation (Dual Mode)  
    3. What-If Simulation  
    4. Cloud-Safe Logging  
    5. Streamlit Deployment  
    """)

# ==========================================
# SIDEBAR CHATBOT
# ==========================================
st.sidebar.title("💬 Clinical AI Assistant")

question = st.sidebar.text_input("Ask medical question")

if st.sidebar.button("Ask"):
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Answer only medical-related questions."},
            {"role": "user", "content": question}
        ],
    )
    st.sidebar.write(response.choices[0].message.content)

st.markdown("---")
st.info("⚠ Educational use only.")