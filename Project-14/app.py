import streamlit as st
import joblib
import pandas as pd

# Load Model
model = joblib.load("model.pkl")

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")

st.write("Enter the patient's details")

age = st.number_input("Age", min_value=1, max_value=120, value=45)

sex = st.selectbox(
    "Sex",
    ["Female", "Male"]
)

cp = st.number_input("Chest Pain Type (cp)", min_value=0, max_value=3, value=0)

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol",
    min_value=100,
    max_value=600,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ["No", "Yes"]
)

restecg = st.number_input(
    "Rest ECG",
    min_value=0,
    max_value=2,
    value=0
)

thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.number_input(
    "Slope",
    min_value=0,
    max_value=2,
    value=1
)

ca = st.number_input(
    "Number of Major Vessels",
    min_value=0,
    max_value=4,
    value=0
)

thal = st.number_input(
    "Thal",
    min_value=0,
    max_value=3,
    value=2
)

# Convert Selectbox Values

sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

if st.button("Predict"):

    data = pd.DataFrame([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]],
    columns=[
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ])

    prediction = model.predict(data)

    if prediction[0] == 0:
        st.success("✅ Heart Disease Detected")
    else:
        st.success("💚 No Heart Disease Detected")
