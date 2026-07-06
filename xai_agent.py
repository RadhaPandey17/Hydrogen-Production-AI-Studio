"""
==========================================================
Hydrogen Production AI Studio (2026)

Explainable AI Agent

==========================================================
"""

import joblib
import pandas as pd

from config import MODEL_PATH


class XAIAgent:

    def __init__(self):

        self.model = joblib.load(MODEL_PATH)

    # --------------------------------------------------------

    def explain(self, scaled_data, feature_names):

        # Placeholder feature importance so that
        # AI Report and PDF generation continue working.

        feature_importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance": [0.0] * len(feature_names)

        })

        feature_importance = feature_importance.sort_values(

            by="Feature"

        )

        return {

            "feature_importance": feature_importance,

            "shap_values": None

        }
