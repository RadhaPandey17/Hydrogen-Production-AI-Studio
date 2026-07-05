"""
==========================================================
Explainable AI Agent

Uses SHAP

Returns

• SHAP Values

• Feature Importance

• Waterfall

• Summary

==========================================================
"""

import joblib
import shap
import numpy as np
import pandas as pd

from config import *


class XAIAgent:

    def __init__(self):

        model = joblib.load(MODEL_PATH)

        try:

            self.model = model.estimators_[1]

        except Exception:

            self.model = model

        self.explainer = shap.TreeExplainer(

            self.model

        )

    # --------------------------------------------------------

    def explain(self, scaled_data, feature_names):

        shap_values = self.explainer.shap_values(

            scaled_data

        )

        importance = np.abs(

            shap_values

        ).mean(axis=0)

        feature_importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance": importance

        })

        feature_importance = feature_importance.sort_values(

            "Importance",

            ascending=False

        )

        return {

            "shap_values": shap_values,

            "feature_importance": feature_importance

        }
