"""
==========================================================
Hydrogen Production AI Studio (2026)

Gemini Report Agent

Uses

• Gemini API
• SHAP Result
• Prediction

Generates AI Sustainability Report

==========================================================
"""

from pathlib import Path

from config import GEMINI_MODEL


class ReportAgent:

    def __init__(self):
        from pathlib import Path
        from config import PROMPT_PATH, GEMINI_MODEL
        class ReportAgent:
            def __init__(self):
                self.model = GEMINI_MODEL
                print("PROMPT PATH =", PROMPT_PATH)
                if not PROMPT_PATH.exists():
                    raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")
                    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
                        self.prompt_template = f.read()

    # ----------------------------------------------------

    def generate_report(
        self,
        prediction_result,
        feature_importance
    ):

        top_features = feature_importance.head(5)

        shap_text = ""

        for _, row in top_features.iterrows():

            shap_text += (
                f"{row['Feature']} : "
                f"{round(row['Importance'],4)}\n"
            )

        prompt = f"""

{self.system_prompt}

==================================================

Prediction

Hydrogen Production

{prediction_result["Hydrogen_Output"]} kg/day

Estimated CO2 Emission

{prediction_result["CO2_Emission"]} kg CO₂-eq/kg H₂

Latitude

{prediction_result["Latitude"]}

Longitude

{prediction_result["Longitude"]}

==================================================

Top SHAP Features

{shap_text}

==================================================

Generate a professional sustainability report.

"""

        response = GEMINI_MODEL.generate_content(prompt)

        return response.text
