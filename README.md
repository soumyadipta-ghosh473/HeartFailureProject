\# Heart Failure Prediction with LLM-Based Clinical Explanation



\## Overview

This project implements a hybrid ML + LLM pipeline for heart failure risk prediction.



\### Architecture

1\. Machine Learning Model (XGBoost) predicts risk

2\. Large Language Model (Groq - Llama 3.3 70B Versatile) generates clinical explanation

3\. Streamlit provides interactive interface

4\. CI/CD implemented using GitHub Actions

5\. Docker used for containerization



\## LLMOps Components

\- Prompt Engineering

\- Secure API key handling using environment variables

\- ML + LLM orchestration

\- Automated CI pipeline on push

\- Docker image build automation



\## Deployment

Local deployment via Streamlit.



\## Security

API keys are managed using environment variables and never committed to source control.

