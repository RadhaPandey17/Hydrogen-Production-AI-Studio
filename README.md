# ⚡ Hydrogen Production AI Studio

Hydrogen Production AI Studio is an AI-powered web application developed to predict hydrogen production using Machine Learning and Life Cycle Assessment (LCA). The platform combines predictive analytics, Explainable AI (SHAP), and AI-generated sustainability reports to support data-driven decision-making for sustainable hydrogen production pathways.

---

## 🚀 Features

- 🤖 Hydrogen Production Prediction using a trained Voting Regressor
- 🌍 Life Cycle Assessment (LCA)-based sustainability analysis
- 📍 Location-based prediction using Latitude & Longitude
- 🔍 Explainable AI with SHAP feature importance
- 🧠 AI-generated sustainability reports using Google Gemini
- 📄 Downloadable PDF reports
- 🎨 Interactive Streamlit dashboard with a clean user interface

---

## 🛠 Technologies Used

- Python
- Streamlit
- Scikit-learn
- XGBoost
- Random Forest
- Ridge Regression
- SHAP (Explainable AI)
- Plotly
- Google Gemini API
- ReportLab
- Pandas & NumPy

---

## 📂 Project Structure

```text
Hydrogen_Production_AI_Studio/

├── app.py
├── prediction_agent.py
├── report_agent.py
├── xai_agent.py
├── config.py
├── pdf_generator.py
├── requirements.txt
├── README.md
├── Hydrogen_LCA_Final_Preprocessed.csv
├── best_final_ensemble_model.pkl
├── scaler.pkl
├── feature_names.pkl
├── report_prompt.txt
└── .env
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/RadhaPandey17/Hydrogen-Production-AI-Studio.git
cd Hydrogen-Production-AI-Studio
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file:

```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

Run the application:

```bash
streamlit run app.py
```

---

## 📊 Workflow

1. Enter Latitude and Longitude.
2. The system identifies the nearest location from the backend dataset.
3. Features are automatically extracted and preprocessed.
4. The trained ML model predicts hydrogen production.
5. SHAP explains the prediction.
6. Google Gemini generates an AI-based sustainability report.
7. The report can be downloaded as a PDF.

---

## 🎯 Applications

- Sustainable Hydrogen Production
- Life Cycle Assessment (LCA)
- Environmental Decision Support
- Renewable Energy Planning
- Explainable AI Research

---

## 👩‍💻 Developer

**Radha Pandey**

Spark Research Intern

**SPARK Internship – IIT Roorkee**

---

## 📜 License

This project is developed for academic and research purposes.
