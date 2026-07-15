import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

st.title("Diabetes Prediction System")

pregnancies = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bloodpressure = st.number_input("Blood Pressure")
skinthickness = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):
    data = np.array([[pregnancies, glucose, bloodpressure,
                      skinthickness, insulin, bmi, dpf, age]])

    prediction = model.predict(data)

    st.success(f"Prediction : {prediction[0]:.2f}")