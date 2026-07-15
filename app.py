import streamlit as st
import joblib
import numpy as np

# Load Model
model = joblib.load("decision_tree_model.pkl")

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient's details below to predict the presence of heart disease.")

# Input Fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.number_input("Sex (0 = Female, 1 = Male)", min_value=0, max_value=1, value=1)
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0)
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol", value=200)
fbs = st.number_input("Fasting Blood Sugar (0/1)", min_value=0, max_value=1, value=0)
restecg = st.number_input("Resting ECG (0-2)", min_value=0, max_value=2, value=0)
thalach = st.number_input("Maximum Heart Rate", value=150)
exang = st.number_input("Exercise Induced Angina (0/1)", min_value=0, max_value=1, value=0)
oldpeak = st.number_input("Oldpeak", value=1.0)
slope = st.number_input("Slope (0-2)", min_value=0, max_value=2, value=1)
ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=0)
thal = st.number_input("Thal (0-3)", min_value=0, max_value=3, value=2)

# Prediction
if st.button("Predict"):
    data = np.array([[age, sex, cp, trestbps, chol,
                      fbs, restecg, thalach, exang,
                      oldpeak, slope, ca, thal]])

    prediction = model.predict(data)[0]

    st.write("Prediction:", prediction)

    if prediction == 1:
        st.success("✅ No Heart Disease Detected")
    else:
        st.error("⚠️ Heart Disease Detected")