import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("Salary_Prediction_Model.pkl", "rb"))

# Title
st.title("💼 Salary Prediction App")

st.write("Enter Years of Experience to predict Salary")

# User Input
experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.1
)

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict([[experience]])

    st.success(f"Predicted Salary: ₹ {prediction[0]:,.2f}")