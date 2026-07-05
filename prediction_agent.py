"""
==========================================================
Hydrogen Production AI Studio (2026)

Prediction Agent

Workflow

Latitude
Longitude

↓

Nearest Dataset Location

↓

Extract Remaining Features

↓

Feature Engineering

↓

Scaling

↓

Prediction

↓

Hydrogen Output
CO₂ Emission
==========================================================
"""

import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from config import DATASET_PATH

from config import * 


class PredictionAgent:

    def __init__(self):

        self.dataset = pd.read_csv(DATASET_PATH)

        self.model = joblib.load(MODEL_PATH)

        self.scaler = joblib.load(SCALER_PATH)

        self.required_features = joblib.load(FEATURE_PATH)

    # --------------------------------------------------------

    def nearest_location(self, latitude, longitude):

        df = self.dataset.copy()

        distance = (
            (df["Latitude"] - latitude) ** 2
            +
            (df["Longitude"] - longitude) ** 2
        )

        nearest_index = distance.idxmin()

        return df.loc[nearest_index].copy()

    # --------------------------------------------------------

    def prepare_features(self, row):

        feature_row = row.copy()

        remove_cols = [

            "Paper_Citation",
            "Location",
            "Latitude",
            "Longitude",
            "Hydrogen_Output_kg_day",
            "Hydrogen_Output_log"

        ]

        for col in remove_cols:

            if col in feature_row.index:

                feature_row = feature_row.drop(col)

        df = pd.DataFrame([feature_row])

        df = pd.get_dummies(

            df,

            columns=[
                "Production_Pathway",
                "Power_Source"
            ],

            drop_first=False

        )

        df.columns = df.columns.str.replace(

            r"[\[\]<>]",

            "_",

            regex=True

        )

        for feature in self.required_features:

            if feature not in df.columns:

                df[feature] = 0

        df = df[self.required_features]

        scaled = self.scaler.transform(df)

        return scaled

    # --------------------------------------------------------

    def predict(self, latitude, longitude):

        nearest = self.nearest_location(

            latitude,

            longitude

        )

        scaled = self.prepare_features(nearest)

        prediction_log = self.model.predict(scaled)[0]

        hydrogen = np.expm1(prediction_log)

        co2 = nearest["LCA_GWP_kg_CO2_eq_per_kg_H2"]

        return {

            "Location": nearest["Location"],

            "Latitude": nearest["Latitude"],

            "Longitude": nearest["Longitude"],

            "Hydrogen_Output": round(float(hydrogen), 2),

            "CO2_Emission": round(float(co2), 2),

            "Feature_Row": nearest,

            "Scaled_Data": scaled

        }
