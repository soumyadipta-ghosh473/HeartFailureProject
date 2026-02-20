import streamlit as st
import joblib
import numpy as np
import os
from groq import Groq
import datetime
import csv
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Heart Failure AI System",
    page_icon="❤️",
    layout="wide"
)

# ==========================================
# PROFESSIONAL THEME
# ==========================================
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.stApp {
    background: linear-gradient(135deg, #1d4350, #2c5364);
    color: #f5f5f5;
}
.stButton>button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
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
# SYSTEM STATUS
# ==========================================
PROMPT_VERSION = "v6.0"
MODEL_VERSION = "HeartFailure-XGB-v1"
LLM_MODEL = "llama-3.3-70b-versatile"

st.markdown("""
## ❤️ AI-Powered Heart Failure Risk Assessment System
### Hybrid ML + LLM Clinical Intelligence Platform
""")

st.markdown("### ⚙ System Status")
st.markdown(f"""
- **ML Model Version:** {MODEL_VERSION}
- **Prompt Version:** {PROMPT_VERSION}
- **LLM Model:** {LLM_MODEL}
- **Logging:** Enabled
- **Deployment:** Streamlit Cloud
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
# LLMOPS ARCHITECTURE
# ==========================================
with st.expander("🔎 View LLMOps Architecture"):
    st.markdown("""
    **Architecture Layers:**

    1️⃣ ML Prediction Layer – XGBoost classification  
    2️⃣ LLM Reasoning Layer – Groq-hosted LLaMA model  
    3️⃣ Monitoring Layer – CSV logging with prompt versioning  
    4️⃣ Deployment Layer – Streamlit Cloud  
    5️⃣ CI/CD – GitHub Actions automated pipeline  
    """)

# ==========================================
# INPUT SECTION
# ==========================================
st.markdown("### 📋 Patient Clinical Parameters")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 1, 120)
    Sex = st.number_input("Sex (0=Female,1=Male)", 0, 1)
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
# CHAT MEMORY INIT
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

    risk_text = "High Risk of Heart Failure" if prediction[0] == 1 else "Low Risk"

    st.metric("Prediction", risk_text)
    st.metric("Confidence", f"{confidence_percentage:.2f}%")

    # ======================================
    # CIRCULAR GAUGE
    # ======================================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_percentage,
        title={'text': "Risk Probability (%)"},
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

    # ======================================
    # LOGGING
    # ======================================
    log_file = "prediction_logs.csv"
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Prediction", "Confidence"])
        writer.writerow([datetime.datetime.now(), risk_text, confidence_percentage])

    # ======================================
    # DOWNLOAD LOG
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
    You are a medical AI assistant.

    Patient details:
    Age: {Age}
    Cholesterol: {Cholesterol}
    Blood Pressure: {RestingBP}
    Max HR: {MaxHR}

    Prediction: {risk_text}
    Confidence: {confidence_percentage:.2f}%

    Provide a clear medical explanation.
    """

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": "You are a medical AI assistant."},
                  {"role": "user", "content": prompt}],
    )

    explanation = response.choices[0].message.content
    st.markdown("## 🧠 AI Clinical Interpretation")
    st.write(explanation)

    # ======================================
    # PDF REPORT
    # ======================================
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.drawString(50, 750, "Heart Failure Risk Report")
    pdf.drawString(50, 720, f"Prediction: {risk_text}")
    pdf.drawString(50, 700, f"Confidence: {confidence_percentage:.2f}%")
    pdf.drawString(50, 680, f"Date: {datetime.datetime.now()}")
    pdf.save()

    buffer.seek(0)

    st.download_button(
        label="📄 Download PDF Report",
        data=buffer,
        file_name="heart_risk_report.pdf",
        mime="application/pdf"
    )

# ==========================================
# SIDEBAR CHATBOT (MEDICAL-ONLY + MEMORY)
# ==========================================
st.sidebar.title("💬 Clinical AI Assistant")

MEDICAL_KEYWORDS = [
    "heart", "cardio", "blood", "pressure", "cholesterol",
    "disease", "pain", "symptom", "treatment", "diagnosis",
    "medicine", "medical", "health", "exercise", "risk",
    "ecg", "bp", "pulse", "therapy"
]

def is_medical_query(query):
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in MEDICAL_KEYWORDS)

user_question = st.sidebar.text_input("Ask a medical question:")

if st.sidebar.button("Ask AI"):

    if not user_question.strip():
        st.sidebar.warning("Please enter a question.")

    elif not is_medical_query(user_question):
        st.sidebar.error("❌ This assistant only answers medical-related questions.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        system_prompt = """
        You are a strict medical assistant.
        Only answer healthcare-related questions.
        Do not provide diagnosis or prescriptions.
        """

        messages = [{"role": "system", "content": system_prompt}] + st.session_state.chat_history

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
        )

        answer = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Display Chat History
for msg in st.session_state.chat_history:
    role = "You" if msg["role"] == "user" else "AI"
    st.sidebar.markdown(f"**{role}:** {msg['content']}")

# ==========================================
# DISCLAIMER
# ==========================================
st.markdown("---")
st.info("⚠ Educational use only. Not a medical diagnosis tool.")