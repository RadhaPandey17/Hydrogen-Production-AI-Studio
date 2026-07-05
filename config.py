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

load_dotenv(PROJECT_ROOT / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")
# ==========================
# FILE PATHS
# ==========================

DATASET_PATH = PROJECT_ROOT / "Hydrogen_LCA_Final_Preprocessed.csv"

MODEL_PATH = PROJECT_ROOT / "best_final_ensemble_model.pkl"

SCALER_PATH = PROJECT_ROOT / "scaler.pkl"

FEATURES_PATH = PROJECT_ROOT / "feature_names.pkl"

PROMPT_PATH = PROJECT_ROOT / "report_prompt.txt"

# ==========================
# OUTPUTS
# ==========================

OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

OUTPUT_FOLDER.mkdir(exist_ok=True)

REPORT_FOLDER = OUTPUT_FOLDER / "reports"

REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
