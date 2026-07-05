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

        prompt_file = (
            Path(__file__)
            .resolve()
            .parent.parent
            / "prompts"
            / "report_prompt.txt"
        )

        with open(prompt_file, "r", encoding="utf-8") as f:

            self.system_prompt = f.read()

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
