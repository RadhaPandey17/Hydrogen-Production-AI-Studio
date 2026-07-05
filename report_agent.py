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
You are an AI sustainability expert.

Generate a professional report containing:

1. Predicted Hydrogen Production
2. Estimated CO₂ Emission
3. SHAP Interpretation
4. Sustainability Insights
5. Recommendations

Keep the report concise and professional.
"""

    # -------------------------------------------------

    def generate_report(self, prediction_result, feature_importance):

        top_features = feature_importance.head(5)

        shap_text = ""

        for _, row in top_features.iterrows():
            shap_text += f"{row['Feature']} : {round(row['Importance'],4)}\n"

        prompt = f"""
{self.system_prompt}

Prediction

Hydrogen Production:
{prediction_result['Hydrogen_Output']} kg/day

Estimated CO₂:
{prediction_result['CO2_Emission']} kg CO₂-eq/kg H₂

Latitude:
{prediction_result['Latitude']}

Longitude:
{prediction_result['Longitude']}

Top SHAP Features:

{shap_text}
"""

        response = self.model.generate_content(prompt)

        return response.text
