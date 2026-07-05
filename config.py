from pathlib import Path
from dotenv import load_dotenv
import os
import google.generativeai as genai

# ===================================================
# PROJECT ROOT
# ===================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ===================================================
# LOAD ENV
# ===================================================

load_dotenv(PROJECT_ROOT / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")

# ===================================================
# PATHS
# ===================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

DATASET_PATH = PROJECT_ROOT / "Hydrogen_LCA_Final_Preprocessed.csv"

MODEL_PATH = PROJECT_ROOT / "best_final_ensemble_model.pkl"

SCALER_PATH = PROJECT_ROOT / "scaler.pkl"

FEATURES_PATH = PROJECT_ROOT / "feature_names.pkl"

PROMPT_PATH = PROJECT_ROOT / "report_prompt.txt"

REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

# ===================================================
# APP
# ===================================================

APP_NAME = "Hydrogen Production AI Studio"

APP_VERSION = "2026.1"
