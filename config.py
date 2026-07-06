from pathlib import Path
from dotenv import load_dotenv
import os
import google.generativeai as genai

# ==========================
# PROJECT ROOT
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent

# ==========================
# LOAD ENV
# ==========================

import streamlit as st

load_dotenv(PROJECT_ROOT / ".env")

GOOGLE_API_KEY = (
    st.secrets.get("GOOGLE_API_KEY")
    if "GOOGLE_API_KEY" in st.secrets
    else os.getenv("GOOGLE_API_KEY")
)
print("API KEY =", GOOGLE_API_KEY)

genai.configure(api_key=GOOGLE_API_KEY)

GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")
# ==========================
# FILE PATHS
# ==========================

DATASET_PATH = PROJECT_ROOT / "Hydrogen_LCA_Final_Preprocessed.csv"

MODEL_PATH = PROJECT_ROOT / "best_final_ensemble_model.pkl"

SCALER_PATH = PROJECT_ROOT / "scaler.pkl"

FEATURE_PATH = PROJECT_ROOT / "feature_names.pkl"

PROMPT_PATH = PROJECT_ROOT / "report_prompt.txt"

# ==========================
# OUTPUTS
# ==========================

OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

OUTPUT_FOLDER.mkdir(exist_ok=True)

REPORT_FOLDER = OUTPUT_FOLDER / "reports"

REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
# ==========================
# APPLICATION INFO
# ==========================

APP_NAME = "Hydrogen Production AI Studio"

APP_VERSION = "2026.1"
