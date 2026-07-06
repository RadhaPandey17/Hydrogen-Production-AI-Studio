import joblib
import shap
import pandas as pd

from config import MODEL_PATH


class XAIAgent:

    def __init__(self):

        self.model = joblib.load(MODEL_PATH)

        self.explainer = shap.Explainer(self.model.predict)

    def explain(self, scaled_data, feature_names):

        shap_values = self.explainer(scaled_data)

        importance = abs(shap_values.values).mean(axis=0)

        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })

        feature_importance = feature_importance.sort_values(
            by="Importance",
            ascending=False
        )

        return {
            "feature_importance": feature_importance,
            "shap_values": shap_values
        }
