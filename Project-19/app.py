import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Milk Quality Prediction",
    page_icon="🥛",
    layout="centered"
)

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = load_model("milk_quality_ann.keras")
scaler = joblib.load("scaler.pkl")

st.title("🥛 Milk Quality Prediction using ANN")
st.write("Enter the milk parameters below and click Predict.")

# -----------------------------
# Inputs
# -----------------------------
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.6)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=0,
    max_value=100,
    value=35
)

taste = st.selectbox(
    "Taste",
    ["Good", "Bad"]
)

odor = st.selectbox(
    "Odor",
    ["Good", "Bad"]
)

fat = st.selectbox(
    "Fat",
    ["High", "Low"]
)

turbidity = st.selectbox(
    "Turbidity",
    ["High", "Low"]
)

colour = st.number_input(
    "Colour",
    min_value=200,
    max_value=300,
    value=255
)

# -----------------------------
# Convert Inputs
# -----------------------------
taste = 1 if taste == "Good" else 0
odor = 1 if odor == "Good" else 0
fat = 1 if fat == "High" else 0
turbidity = 1 if turbidity == "High" else 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Milk Quality"):

    sample = np.array([[
        ph,
        temperature,
        taste,
        odor,
        fat,
        turbidity,
        colour
    ]])

    sample = scaler.transform(sample)

    prediction = model.predict(sample, verbose=0)

    pred = np.argmax(prediction)

    labels = {
        0: "High",
        1: "Low",
        2: "Medium"
    }

    result = labels[pred]

    st.success(f"Predicted Grade : {result}")

    if result == "High":
        st.success("🟢 Excellent Milk Quality")

    elif result == "Medium":
        st.warning("🟡 Medium Milk Quality")

    else:
        st.error("🔴 Low Milk Quality")
