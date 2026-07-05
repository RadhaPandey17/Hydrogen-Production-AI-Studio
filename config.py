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

DATASET_PATH = PROJECT_ROOT / "data" / "Hydrogen_LCA_Final_Preprocessed.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "best_final_ensemble_model.pkl"

SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

FEATURE_PATH = PROJECT_ROOT / "models" / "feature_names.pkl"

PROMPT_PATH = PROJECT_ROOT / "prompts" / "report_prompt.txt"

OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

REPORT_FOLDER = OUTPUT_FOLDER / "reports"

REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

# ===================================================
# APP
# ===================================================

APP_NAME = "Hydrogen Production AI Studio"

APP_VERSION = "2026.1"