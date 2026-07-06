"""
Hydrogen Production AI Studio
Configuration File
Compatible with Streamlit Cloud + Local VS Code
"""

from pathlib import Path
import os

from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# ==========================================================
# LOAD LOCAL .ENV
# ==========================================================

load_dotenv(PROJECT_ROOT / ".env")

# ==========================================================
# GOOGLE GEMINI CONFIGURATION
# ==========================================================

GOOGLE_API_KEY = None

# First try Streamlit Secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

# Then local .env
if GOOGLE_API_KEY is None:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        """
GOOGLE_API_KEY not found.

Local VS Code
-------------
Create a .env file containing

GOOGLE_API_KEY=YOUR_API_KEY

Streamlit Cloud
---------------
App Settings
→ Secrets

GOOGLE_API_KEY="YOUR_API_KEY"
"""
    )

# ==========================================================
# INITIALIZE GEMINI
# ==========================================================

try:

    genai.configure(api_key=GOOGLE_API_KEY)

    GEMINI_MODEL = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

except Exception as e:

    raise RuntimeError(
        f"Failed to initialize Gemini.\n\n{e}"
    )

print("=" * 60)
print("Hydrogen Production AI Studio")
print("Project Root :", PROJECT_ROOT)
print("Gemini Model :", "gemini-2.5-flash")
print("Gemini Loaded Successfully")
print("=" * 60)

# ==========================================================
# FILE PATHS
# ==========================================================

DATASET_PATH = PROJECT_ROOT / "Hydrogen_LCA_Final_Preprocessed.csv"

MODEL_PATH = PROJECT_ROOT / "best_final_ensemble_model.pkl"

SCALER_PATH = PROJECT_ROOT / "scaler.pkl"

FEATURE_PATH = PROJECT_ROOT / "feature_names.pkl"

PROMPT_PATH = PROJECT_ROOT / "report_prompt.txt"

# ==========================================================
# OUTPUT FOLDERS
# ==========================================================

OUTPUT_FOLDER = PROJECT_ROOT / "outputs"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

REPORT_FOLDER = OUTPUT_FOLDER / "reports"
REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

# ==========================================================
# APP INFO
# ==========================================================

APP_NAME = "Hydrogen Production AI Studio"

APP_VERSION = "2026.1"

AUTHOR = "Radha Pandey"

# ==========================================================
# SETTINGS
# ==========================================================

DEFAULT_SHAP_SAMPLE = 150

RANDOM_STATE = 42

# ==========================================================
# VERIFY FILES
# ==========================================================

required_files = {
    "Dataset": DATASET_PATH,
    "Model": MODEL_PATH,
    "Scaler": SCALER_PATH,
    "Feature Names": FEATURE_PATH,
    "Prompt": PROMPT_PATH,
}

missing_files = []

for name, path in required_files.items():

    if not path.exists():

        missing_files.append(f"{name}: {path}")

if missing_files:

    print("=" * 60)
    print("WARNING : Missing Files")
    print("=" * 60)

    for item in missing_files:
        print(item)

    print("=" * 60)

else:

    print("All required files found.")
