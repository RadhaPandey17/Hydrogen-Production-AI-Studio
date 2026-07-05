# ==========================================================
# Hydrogen Production AI Studio (2026)
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

from prediction_agent import PredictionAgent
from xai_agent import XAIAgent
from report_agent import ReportAgent

from pdf_generator import PDFGenerator
from config import APP_NAME, APP_VERSION

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
"""
<style>

html, body, [class*="css"]{
    font-family: "Segoe UI";
    background:#f5f7fb;
}

.main{
    background:#f5f7fb;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

section[data-testid="stSidebar"]{
    background:#0F172A;
}

section[data-testid="stSidebar"] *{
    color:white;
}

.metric-card{

background:#1E293B;

color:white;

padding:25px;

border-radius:18px;

box-shadow:0 10px 20px rgba(0,0,0,.3);

} .metric-card h2{

color:white;

}

.metric-card h3{

color:#60A5FA;

}

.metric-card p{

color:#CBD5E1;

}

.prediction-card{

    background:linear-gradient(
        135deg,
        #2563EB,
        #1E3A8A
    );

    color:white;

    border-radius:20px;

    padding:30px;

}

.footer{

    text-align:center;

    color:gray;

    margin-top:50px;

}

</style>
""",
unsafe_allow_html=True
)

# ==========================================================
# INITIALIZE AGENTS
# ==========================================================

@st.cache_resource

def load_prediction():

    return PredictionAgent()


@st.cache_resource

def load_xai():

    return XAIAgent()


@st.cache_resource

def load_report():

    return ReportAgent()


@st.cache_resource

def load_pdf():

    return PDFGenerator()


prediction_agent = load_prediction()

xai_agent = load_xai()

report_agent = load_report()

pdf_generator = load_pdf()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("⚡ Hydrogen AI Studio")

page = st.sidebar.radio(

    "Navigation",

    [

        "Dashboard",

        "Prediction",

        "AI Report",

        "About"

    ]

)

st.sidebar.markdown("---")

st.sidebar.info(

f"""
Version

{APP_VERSION}

Machine Learning

Voting Regressor

Explainability

SHAP

Powered by

Google Gemini
"""
)

# ==========================================================
# SESSION STATE
# ==========================================================

if "prediction" not in st.session_state:

    st.session_state.prediction = None

if "feature_importance" not in st.session_state:

    st.session_state.feature_importance = None

if "report" not in st.session_state:

    st.session_state.report = None
    # ==========================================================
# DASHBOARD
# ==========================================================

if page == "Dashboard":

    st.title("⚡ Hydrogen Production AI Studio")

    st.caption(
        "Machine Learning Assisted Life Cycle Assessment of Hydrogen Production Pathways"
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="metric-card">

            <h3>🤖 AI Model</h3>

            <h2>Voting Regressor</h2>

            <p>Random Forest + Ridge + XGBoost</p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="metric-card">

            <h3>🌍 Explainability</h3>

            <h2>SHAP AI</h2>

            <p>Model Interpretation</p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="metric-card">

            <h3>📄 AI Report</h3>

            <h2>Gemini</h2>

            <p>Professional Sustainability Report</p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## Workflow")

    st.info(
        """
Latitude + Longitude

⬇

Nearest Dataset Match

⬇

Feature Extraction

⬇

Prediction

⬇

SHAP Explanation

⬇

AI Report

⬇

PDF Download
"""
    )

# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "Prediction":

    st.title("⚡ Hydrogen Prediction")

    st.write(
        "Enter Latitude and Longitude. "
        "The remaining parameters will be retrieved automatically."
    )

    st.markdown("---")

    with st.container(border=True):

        st.subheader("📍 Location Coordinates")

        col1, col2 = st.columns(2)

        with col1:
            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=23.500,
                format="%.6f"
            )

        with col2:
            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=78.900,
                format="%.6f"
            )

        predict_btn = st.button(
            "🚀 Predict Hydrogen Production",
            use_container_width=True
        )

    # ======================================================
    # PREDICTION PIPELINE
    # ======================================================

    if predict_btn:

        with st.spinner("Running AI Prediction..."):

            try:

                # Prediction
                prediction_result = prediction_agent.predict(
                    latitude=latitude,
                    longitude=longitude
                )

                st.session_state.prediction = prediction_result

                # SHAP
                xai_result = xai_agent.explain(
                    prediction_result["Scaled_Data"],
                    prediction_agent.required_features
                )

                st.session_state.feature_importance = xai_result["feature_importance"]

                # AI REPORT
                report = report_agent.generate_report(
                    prediction_result,
                    st.session_state.feature_importance
                )

                st.session_state.report = report

                st.success("Prediction Completed Successfully!")

            except Exception as e:

                import traceback

                st.error("Prediction / AI Report Failed")

                st.exception(e)

                st.code(traceback.format_exc())

    # ======================================================
    # SHOW RESULTS
    # ======================================================

    if st.session_state.prediction is not None:

        result = st.session_state.prediction

        st.markdown("---")

        st.subheader("Prediction Results")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Hydrogen Production",
                f"{result['Hydrogen_Output']} kg/day"
            )

        with c2:
            st.metric(
                "Estimated CO₂",
                f"{result['CO2_Emission']} kg CO₂-eq"
            )

        with c3:
            score = round(
                100 - min(result["CO2_Emission"] * 5, 100),
                1
            )

            st.metric(
                "Sustainability Score",
                f"{score} %"
            )

        st.markdown("---")

        st.subheader("Matched Dataset Location")

        info_df = result["Feature_Row"].to_frame()
        info_df.columns = ["Value"]

        st.dataframe(
            info_df,
            use_container_width=True
        )

        # ======================================================
        # SHAP FEATURE IMPORTANCE
        # ======================================================

        if st.session_state.feature_importance is not None:

            st.markdown("---")

            st.subheader("🔍 Top Factors Affecting Prediction")

            importance = (
                st.session_state.feature_importance
                .head(10)
                .sort_values("Importance")
            )

            fig = px.bar(
                importance,
                x="Importance",
                y="Feature",
                orientation="h",
                text="Importance",
                title="SHAP Feature Importance"
            )

            fig.update_layout(
                height=500,
                template="plotly_white",
                title_x=0.5,
                xaxis_title="SHAP Importance",
                yaxis_title="Feature"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )
# ==========================================================
# AI REPORT PAGE
# ==========================================================

elif page == "AI Report":

    st.title("🤖 AI Sustainability Report")

    if st.session_state.report is None:

        st.warning(

            "Please generate a prediction first."

        )

    else:

        st.markdown(

            st.session_state.report

        )

        pdf_buffer = pdf_generator.generate(

            st.session_state.prediction,

            st.session_state.report

        )

        st.download_button(

            label="📄 Download PDF Report",

            data=pdf_buffer,

            file_name="Hydrogen_AI_Report.pdf",

            mime="application/pdf",

            use_container_width=True

        )

# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "About":

    st.title("ℹ About Hydrogen Production AI Studio")

    st.markdown("""

### Hydrogen Production AI Studio (2026)

Hydrogen Production AI Studio is an AI-powered application that predicts hydrogen production using a pre-trained Machine Learning model.

The system automatically retrieves the nearest dataset entry using only latitude and longitude, performs feature engineering, predicts hydrogen production, estimates environmental impact, explains the prediction using SHAP, and generates an AI-powered sustainability report.

---

### Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Random Forest
- Ridge Regression
- XGBoost
- Voting Regressor
- SHAP
- Plotly
- Google Gemini
- ReportLab

---

### Features

✅ Hydrogen Prediction

✅ CO₂ Estimation

✅ SHAP Explainability

✅ AI Sustainability Report

✅ PDF Report Generation

---

### Developed By

Radha Pandey

Spark Research Internship

Supervisor : Prof Pratham Arora , Department of Hydro And Renewable Energy
IIT Roorkee

Hydrogen Production AI Studio (2026)

""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(

"""

<div class="footer">

Hydrogen Production AI Studio • Version 2026.1

<br>

Powered by Machine Learning • SHAP • Google Gemini

</div>

""",

unsafe_allow_html=True

)
