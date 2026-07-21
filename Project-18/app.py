import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# -----------------------------
# Load Model
# -----------------------------
model = load_model("wine_quality_ann.keras")

st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

st.title("🍷 Wine Quality Prediction using ANN")
st.write("Enter the wine parameters below and click Predict.")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
    volatile_acidity = st.number_input("Volatile Acidity", value=0.70)
    citric_acid = st.number_input("Citric Acid", value=0.00)
    residual_sugar = st.number_input("Residual Sugar", value=1.9)
    chlorides = st.number_input("Chlorides", value=0.076)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)

with col2:
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
    density = st.number_input("Density", value=0.9978)
    pH = st.number_input("pH", value=3.51)
    sulphates = st.number_input("Sulphates", value=0.56)
    alcohol = st.number_input("Alcohol", value=9.4)

if st.button("🔮 Predict Wine Quality"):

    data = np.array([[fixed_acidity,
                      volatile_acidity,
                      citric_acid,
                      residual_sugar,
                      chlorides,
                      free_sulfur_dioxide,
                      total_sulfur_dioxide,
                      density,
                      pH,
                      sulphates,
                      alcohol]])

    prediction = model.predict(data, verbose=0)[0][0]

    st.success(f"Predicted Wine Quality: {prediction:.2f}")

    if prediction < 5:
        st.error("🍷 Poor Quality Wine")
    elif prediction < 7:
        st.warning("🍷 Average Quality Wine")
    else:
        st.success("🍷 Excellent Quality Wine")
