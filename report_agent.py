"""
Hydrogen Production AI Studio
Gemini Report Agent
Compatible with google-genai SDK
"""

from google import genai

from config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    PROMPT_PATH,
)


class ReportAgent:

    def __init__(self):

        # ----------------------------------------
        # Load Prompt
        # ----------------------------------------

        if PROMPT_PATH.exists():

            with open(PROMPT_PATH, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()

        else:

            self.system_prompt = """
You are an AI Sustainability Expert.

Generate a professional sustainability report containing:

1. Executive Summary
2. Predicted Hydrogen Production
3. Estimated CO₂ Emission
4. Key Influencing Factors
5. Sustainability Assessment
6. Recommendations

Write in professional research language.
"""

        # ----------------------------------------
        # Gemini Client
        # ----------------------------------------

        self.client = genai.Client(
            api_key=GOOGLE_API_KEY
        )

    # =======================================================
    # Generate Report
    # =======================================================

    def generate_report(
        self,
        prediction_result,
        feature_importance,
    ):

        # ----------------------------------------

        top_features = feature_importance.head(5)

        shap_text = ""

        for _, row in top_features.iterrows():

            shap_text += (
                f"- {row['Feature']} : "
                f"{float(row['Importance']):.4f}\n"
            )

        # ----------------------------------------

        prompt = f"""
{self.system_prompt}

Prediction Results

Location:
{prediction_result.get("Location","Unknown")}

Latitude:
{prediction_result.get("Latitude","N/A")}

Longitude:
{prediction_result.get("Longitude","N/A")}

Predicted Hydrogen Production:
{prediction_result.get("Hydrogen_Output","N/A")} kg/day

Estimated CO₂ Emission:
{prediction_result.get("CO2_Emission","N/A")} kg CO₂-eq/kg H₂

Top Influencing Features

{shap_text}

Generate a professional sustainability report.
"""

        try:

            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            if response.text:
                return response.text

            return "Gemini returned an empty response."

        except Exception as e:

            return f"""
AI Sustainability Report

Report generation failed.

Reason:
{e}

------------------------------------

Prediction Summary

Hydrogen Production:
{prediction_result.get("Hydrogen_Output","N/A")} kg/day

Estimated CO₂ Emission:
{prediction_result.get("CO2_Emission","N/A")} kg CO₂-eq/kg H₂

Top SHAP Features

{shap_text}
"""
