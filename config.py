"""
Hydrogen Production AI Studio
Configuration File
"""

from pathlib import Path
from dotenv import load_dotenv
import os
import google.generativeai as genai

# =====================================================
# PROJECT ROOT
# =====================================================

PROJECT_ROOT = Path(_file_).resolve().parent

# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv(PROJECT_ROOT / ".env")

# =====================================================
# GOOGLE GEMINI CONFIGURATION
# =====================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. "
        "Please add it to your Render Environment Variables or local .env file."
    )

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Gemini Model
GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")

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
# OUTPUT DIRECTORIES
# =====================================================

OUTPUT_FOLDER = PROJECT_ROOT / "outputs"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

REPORT_FOLDER = OUTPUT_FOLDER / "reports"
REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

# =====================================================
# APPLICATION INFO
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
# VERIFY REQUIRED FILES
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
        print(f" - {item}")
else:
    print("All required files found.")
