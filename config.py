"""
Hydrogen Production AI Studio
Configuration File
Compatible with Local + Render + Streamlit
"""

from pathlib import Path
import os

from dotenv import load_dotenv

# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# ==========================================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv(PROJECT_ROOT / ".env")

# ==========================================================
# GOOGLE GEMINI CONFIGURATION
# ==========================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

if not GOOGLE_API_KEY:
    raise RuntimeError(
        """
GOOGLE_API_KEY not found.

For Local Development:
----------------------
Create a .env file

GOOGLE_API_KEY=YOUR_API_KEY

For Render:
-----------
Dashboard
→ Environment
→ Add Environment Variable

GOOGLE_API_KEY=YOUR_API_KEY

(Optional)

GEMINI_MODEL=gemini-2.5-flash
"""
    )

print("=" * 60)
print("Hydrogen Production AI Studio")
print("Project Root :", PROJECT_ROOT)
print("Gemini Model :", GEMINI_MODEL)
print("Environment Loaded Successfully")
print("=" * 60)

# ==========================================================
# DATASET
# ==========================================================

DATASET_PATH = PROJECT_ROOT / "Hydrogen_LCA_Final_Preprocessed.csv"

# ==========================================================
# MODEL FILES
# ==========================================================

MODEL_PATH = PROJECT_ROOT / "best_final_ensemble_model.pkl"

SCALER_PATH = PROJECT_ROOT / "scaler.pkl"

FEATURE_PATH = PROJECT_ROOT / "feature_names.pkl"

# ==========================================================
# REPORT PROMPT
# ==========================================================

PROMPT_PATH = PROJECT_ROOT / "report_prompt.txt"

# ==========================================================
# OUTPUT FOLDERS
# ==========================================================

OUTPUT_FOLDER = PROJECT_ROOT / "outputs"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

REPORT_FOLDER = OUTPUT_FOLDER / "reports"
REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

# ==========================================================
# APPLICATION INFO
# ==========================================================

APP_NAME = "Hydrogen Production AI Studio"

APP_VERSION = "2026.1"

AUTHOR = "Radha Pandey"

# ==========================================================
# DEFAULT SETTINGS
# ==========================================================

DEFAULT_SHAP_SAMPLE = 150

RANDOM_STATE = 42

# ==========================================================
# VERIFY REQUIRED FILES
# ==========================================================

REQUIRED_FILES = {
    "Dataset": DATASET_PATH,
    "Model": MODEL_PATH,
    "Scaler": SCALER_PATH,
    "Feature Names": FEATURE_PATH,
    "Prompt": PROMPT_PATH,
}

missing_files = []

for name, path in REQUIRED_FILES.items():
    if not path.exists():
        missing_files.append(f"{name}: {path}")

if missing_files:
    print("\n" + "=" * 60)
    print("WARNING: Missing Files")
    print("=" * 60)

    for item in missing_files:
        print(item)

    print("=" * 60)

else:
    print("All required files found.")
