"""
Hydrogen Production AI Studio
Gemini Report Agent
"""

from config import GEMINI_MODEL, PROMPT_PATH


class ReportAgent:

    def __init__(self):

        self.model = GEMINI_MODEL

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

Keep the report concise and professional.
"""

    # ----------------------------------------------------------

    def generate_report(self, prediction_result, feature_importance):

        top_features = feature_importance.head(5)

        shap_text = ""

        for _, row in top_features.iterrows():

            shap_text += (
                f"- {row['Feature']} : "
                f"{round(float(row['Importance']),4)}\n"
            )

        prompt = f"""
{self.system_prompt}

Prediction Results

Location:
{prediction_result['Location']}

Latitude:
{prediction_result['Latitude']}

Longitude:
{prediction_result['Longitude']}

Predicted Hydrogen Production:
{prediction_result['Hydrogen_Output']} kg/day

Estimated CO₂ Emission:
{prediction_result['CO2_Emission']} kg CO₂-eq/kg H₂

Top Influencing Features

{shap_text}

Generate a professional sustainability assessment.
"""

        try:

            response = self.model.generate_content(prompt)

            if hasattr(response, "text"):

                return response.text

            return str(response)

        except Exception as e:

            return f"""
# AI Sustainability Report

Gemini Report Generation Failed.

Reason

{e}

Prediction Summary

Hydrogen Production : {prediction_result['Hydrogen_Output']} kg/day

CO₂ Emission : {prediction_result['CO2_Emission']} kg CO₂-eq/kg H₂

Top Features

{shap_text}
"""
