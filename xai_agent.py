"""
==========================================================
Explainable AI Agent

Uses SHAP

Returns

• SHAP Values
• Feature Importance

==========================================================
"""

import joblib
import shap
import numpy as np
import pandas as pd

from config import MODEL_PATH


class XAIAgent:

    def __init__(self):

        model = joblib.load(MODEL_PATH)

        # If Voting Regressor
        if hasattr(model, "named_estimators_"):

            estimators = model.named_estimators_

            if "RandomForest" in estimators:
                self.model = estimators["RandomForest"]

            elif "randomforestregressor" in estimators:
                self.model = estimators["randomforestregressor"]

            else:
                self.model = list(estimators.values())[0]

        # Single tree model
        else:
            self.model = model

        self.explainer = shap.TreeExplainer(self.model)

    # -----------------------------------------------------

    def explain(self, scaled_data, feature_names):

        shap_values = self.explainer.shap_values(scaled_data)

        # Convert list output to ndarray if required
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        importance = np.abs(shap_values).mean(axis=0)

        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })

        feature_importance = feature_importance.sort_values(
            by="Importance",
            ascending=False
        ).reset_index(drop=True)

        return {
            "shap_values": shap_values,
            "feature_importance": feature_importance
        }
