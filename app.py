import streamlit as st
import joblib
import numpy as np
import os
from groq import Groq
import datetime
import csv
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

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
PROMPT_VERSION = "v11.0"
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
# SESSION LOG INIT
# ==========================================
if "session_logs" not in st.session_state:
    st.session_state.session_logs = []

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
        Oldpeak = st.number_input("Oldpeak (ST Depression)")
        ST_Slope = st.number_input("ST Slope (0–2)", 0, 2)

    # What-If Simulation
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

        # Add to session log
        st.session_state.session_logs.append(risk_text)

        # Write to CSV (safe append)
        file_exists = os.path.isfile("prediction_logs.csv")
        with open("prediction_logs.csv", "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Prediction", "Confidence"])
            writer.writerow([datetime.datetime.now(), risk_text, confidence])

        # Gauge Chart
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

        # LLM Explanation
        if doctor_mode:
            prompt = f"Provide technical medical reasoning for {risk_text} with confidence {confidence:.2f}%."
        else:
            prompt = f"Explain in simple patient-friendly language why the risk is {risk_text}."

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a medical assistant. Only provide medical explanations."},
                {"role": "user", "content": prompt}
            ],
        )

        st.markdown("## 🧠 AI Explanation")
        st.write(response.choices[0].message.content)

# ==========================================
# TAB 2 — ANALYTICS DASHBOARD
# ==========================================
with tab2:

    st.subheader("📊 Risk Distribution")

    high = 0
    low = 0

    # Load CSV if exists
    if os.path.exists("prediction_logs.csv"):
        try:
            df = pd.read_csv("prediction_logs.csv")
            high += (df["Prediction"] == "High Risk").sum()
            low += (df["Prediction"] == "Low Risk").sum()
        except:
            pass

    # Add session logs
    for entry in st.session_state.session_logs:
        if entry == "High Risk":
            high += 1
        elif entry == "Low Risk":
            low += 1

    total = high + low

    st.metric("Total Predictions", total)
    st.metric("High Risk %", round((high / total) * 100 if total > 0 else 0, 2))

    pie_data = pd.DataFrame({
        "Category": ["High Risk", "Low Risk"],
        "Count": [high, low]
    })

    fig = px.pie(
        pie_data,
        values="Count",
        names="Category",
        color="Category",
        color_discrete_map={
            "High Risk": "red",
            "Low Risk": "green"
        }
    )

    fig.update_traces(textinfo="percent+label")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)

    if total == 0:
        st.info("No predictions yet. Generate some in Risk Assessment tab.")

# ==========================================
# TAB 3 — SYSTEM INFO
# ==========================================
with tab3:
    st.markdown("""
    **System Architecture**

    1. Machine Learning Risk Prediction  
    2. LLM Clinical Explanation Layer  
    3. What-If Simulation Engine  
    4. Monitoring & Logging (CSV + Session)  
    5. Streamlit Cloud Deployment  
    """)

# ==========================================
# SIDEBAR CHATBOT (MEDICAL ONLY)
# ==========================================
st.sidebar.title("💬 Clinical AI Assistant")

MEDICAL_KEYWORDS = ["heart", "blood", "disease", "medical", "health", "cholesterol", "bp", "cardio"]

def is_medical(q):
    return any(k in q.lower() for k in MEDICAL_KEYWORDS)

question = st.sidebar.text_input("Ask medical question")

if st.sidebar.button("Ask"):
    if not is_medical(question):
        st.sidebar.error("Only medical-related questions allowed.")
    else:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a medical assistant. Only answer medical questions."},
                {"role": "user", "content": question}
            ],
        )
        st.sidebar.write(response.choices[0].message.content)

st.markdown("---")
st.info("⚠ Educational use only.")