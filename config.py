"""
Hydrogen Production AI Studio
Configuration File
"""

from pathlib import Path
import os

from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

# =====================================================
# PROJECT ROOT
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# =====================================================
# LOAD LOCAL .ENV (FOR VS CODE)
# =====================================================

load_dotenv(PROJECT_ROOT / ".env")

# =====================================================
# GOOGLE API KEY
# =====================================================

GOOGLE_API_KEY = None

# First try Streamlit Secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

# If not found, use local .env
if GOOGLE_API_KEY is None:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Final check
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found.\n"
        "Add it to:\n"
        "1. Streamlit Secrets (Cloud)\n"
        "OR\n"
        "2. Local .env file (VS Code)"
    )

# =====================================================
# CONFIGURE GEMINI
# =====================================================

genai.configure(api_key=GOOGLE_API_KEY)

GEMINI_MODEL = genai.GenerativeModel(
    "gemini-2.5-flash"
)

print("=" * 60)
print("Hydrogen Production AI Studio")
print("Gemini API Loaded Successfully")
print("Using Model : gemini-2.5-flash")
print("=" * 60)

# =====================================================
# FILE PATHS
# =====================================================

DATASET_PATH = PROJECT_ROOT / "Hydrogen_LCA_Final_Preprocessed.csv"

MODEL_PATH = PROJECT_ROOT / "best_final_ensemble_model.pkl"

SCALER_PATH = PROJECT_ROOT / "scaler.pkl"

FEATURE_PATH = PROJECT_ROOT / "feature_names.pkl"

PROMPT_PATH = PROJECT_ROOT / "report_prompt.txt"

# =====================================================
# OUTPUT FOLDERS
# =====================================================

OUTPUT_FOLDER = PROJECT_ROOT / "outputs"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

REPORT_FOLDER = OUTPUT_FOLDER / "reports"
REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

# =====================================================
# APP INFO
# =====================================================

APP_NAME = "Hydrogen Production AI Studio"

APP_VERSION = "2026.1"

AUTHOR = "Radha Pandey"

# =====================================================
# DEFAULT SETTINGS
# =====================================================

DEFAULT_SHAP_SAMPLE = 150

RANDOM_STATE = 42

# =====================================================
# VERIFY FILES
# =====================================================

REQUIRED_FILES = {
    "Dataset": DATASET_PATH,
    "Model": MODEL_PATH,
    "Scaler": SCALER_PATH,
    "Feature Names": FEATURE_PATH,
}

missing = []

for name, path in REQUIRED_FILES.items():
    if not path.exists():
        missing.append(f"{name}: {path}")

if missing:
    print("\nMissing Files:")
    for item in missing:
        print(" -", item)
else:
    print("All required files found.")
