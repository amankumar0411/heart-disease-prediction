import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prediction import HeartDiseasePredictor
import joblib

# Vibrant CSS
st.markdown("""
<style>
    :root {
        --primary: #f72585;
        --secondary: #7209b7;
        --accent: #3a0ca3;
        --light: #fff;
        --dark: #2c2c2c;
        --warning: #f9c74f;
        --success: #43aa8b;
        --info: #4285f4;
    }
    body { background-color: var(--light); color: var(--dark); }
    .stButton button {
        background-color: var(--accent); color: var(--light);
        border: none; border-radius: 8px; font-weight: bold;
        padding: 10px 25px; transition: all 0.2s;
    }
    .stButton button:hover { background: var(--secondary); transform: translateY(-2px); }
    .metric-card {
        background: #e63946; color: white; border-radius: 12px;
        padding: 20px; margin: 10px 0; box-shadow: 0 0 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .risk-low { background: #43aa8b; color: white; }
    .risk-medium { background: #f9c74f; color: var(--dark); }
    .risk-high { background: #f72585; color: white; }
    .risk-result { font-size: 1.5rem; text-align: center; border-radius: 10px; padding: 0.8rem; margin: 1rem 0; }
    .header { font-size: 2.5rem; color: var(--primary); text-align: center; margin-bottom: 1.5rem; font-weight: 700; }
    .card { background: linear-gradient(90deg, #8c2ae9, #4329f4); color: white; border-radius: 12px; padding: 20px; margin: 1rem 0; box-shadow: 0 0 15px rgba(0,0,0,0.1); }
    .section-title { font-size: 1.6rem; color: var(--accent); margin-bottom: 1rem; font-weight: 600; border-bottom: 2px solid var(--accent); display: inline-block; }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.markdown('<div class="header">Heart Disease Prediction System</div>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
  <p>Welcome! Enter patient details below for a single prediction. Explore the dataset and compare model performances below.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------
# Single Prediction
# --------------------------
st.markdown('<div class="section-title">Patient Details & Single Prediction</div>', unsafe_allow_html=True)
with st.form("single_patient_form"):
    age = st.selectbox("Age", list(range(20, 101)))
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.selectbox("Resting Blood Pressure (mm Hg)", list(range(90, 201)))
    chol = st.selectbox("Cholesterol (mg/dl)", list(range(120, 601, 10)))
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.selectbox("Maximum Heart Rate", list(range(60, 221)))
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.select_slider("ST Depression", options=[x*0.1 for x in range(61)], value=1.0)
    slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Major Vessels Colored (0–4)", list(range(5)))
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    model_choice = st.selectbox("Select Model", ["Logistic Regression", "SVM", "KNN", "Random Forest"])
    submitted = st.form_submit_button("🔍 Predict Heart Disease Risk")

if submitted:
    input_data = {
        'age': age,
        'sex': 1 if sex == "Male" else 0,
        'cp': ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp),
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 1 if fbs == "Yes" else 0,
        'restecg': ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg),
        'thalach': thalach,
        'exang': 1 if exang == "Yes" else 0,
        'oldpeak': oldpeak,
        'slope': ["Upsloping", "Flat", "Downsloping"].index(slope),
        'ca': ca,
        'thal': ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1
    }

    predictor = HeartDiseasePredictor()
    result = predictor.predict_single(input_data, model_choice)
    risk_class = "risk-low" if result['risk_level'] == "Low Risk" else "risk-medium" if result['risk_level'] == "Moderate Risk" else "risk-high"
    prediction_text = "Heart Disease Detected" if result['prediction'] == 1 else "No Heart Disease"
    
    st.markdown(f"""
    <div class="metric-card {risk_class}">
        <h3>Prediction Results</h3>
        <p><strong>Risk Level:</strong> {result['risk_level']}</p>
        <p><strong>Result:</strong> {prediction_text}</p>
        <p><strong>Model Used:</strong> {model_choice}</p>
        <p><strong>Disease Probability:</strong> {result['probability_disease']:.1%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = result['probability_disease'] * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Percentage"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#3a0ca3"},
            'steps': [
                {'range': [0, 30], 'color': "#43aa8b"},
                {'range': [30, 70], 'color': "#f9c74f"},
                {'range': [70, 100], 'color': "#f72585"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Data Analysis (EDA + Outliers)
# --------------------------
st.markdown('<div class="section-title">Dataset Exploration</div>', unsafe_allow_html=True)
try:
    df = pd.read_csv('heart.csv', header=None)
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df.columns = cols
    df['target'] = (df['target'] > 0).astype(int)

    stats = pd.DataFrame({
        "Total Patients": [len(df)],
        "Patients with Heart Disease": [df['target'].sum()],
        "Number of features": [len(df.columns)-1]
    }).T.rename(columns={0: "Count"})
    st.markdown(f'<div class="card">{stats.to_html()}</div>', unsafe_allow_html=True)

    


    # Correlation Matrix
    st.subheader("Correlation Matrix")
    corr = df.corr(numeric_only=True)
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Viridis")
    st.plotly_chart(fig_corr, use_container_width=True)

except Exception as e:
    st.error(f"Unable to load dataset for analysis: {e}")

# --------------------------
# Model Comparison (metrics only)
# --------------------------
st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
try:
    scores = joblib.load('models/model_scores.pkl')
    comparison_df = pd.DataFrame(scores).T
    metric_cols = ["accuracy"]
    rename_map = {
        "accuracy": "Accuracy"
    }
    show_df = comparison_df[metric_cols].rename(columns=rename_map)
    st.dataframe(show_df, use_container_width=True)

    for metric in show_df.columns:
        fig = px.bar(
            show_df,
            y=metric,
            title=f"{metric} Comparison",
            color=show_df.index.map({
                'Logistic Regression': '#f72585',
                'SVM': '#3a0ca3',
                'KNN': '#f9c74f',
                'Random Forest': '#43aa8b'
            })
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"Please run 'python src/train_models.py' first to train models. Details: {e}")