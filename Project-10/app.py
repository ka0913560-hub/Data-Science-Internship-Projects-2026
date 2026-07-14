import streamlit as st
import pickle
import numpy as np

# Load Model
model = pickle.load(open("Logistic_Regression_Model.pkl", "rb"))

st.set_page_config(page_title="Car Price Classification", page_icon="🚗")

st.title("🚗 Car Price Classification")
st.write("Enter Car Details to Predict Whether the Car is Expensive or Cheap")

# Inputs
year = st.number_input("Year", min_value=1990, max_value=2025, value=2018)

km_driven = st.number_input("KM Driven", min_value=0, value=50000)

fuel = st.selectbox(
    "Fuel",
    ["Diesel", "Petrol", "LPG", "CNG"]
)

seller = st.selectbox(
    "Seller Type",
    ["Individual", "Dealer", "Trustmark Dealer"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

owner = st.selectbox(
    "Owner",
    [
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car"
    ]
)

mileage = st.number_input("Mileage (km/ltr)", min_value=0.0, value=20.0)

engine = st.number_input("Engine (CC)", min_value=500.0, value=1200.0)

max_power = st.number_input("Max Power", min_value=10.0, value=75.0)

seats = st.number_input("Seats", min_value=2.0, max_value=10.0, value=5.0)

# --------------------------
# Encoding (same as training)
# --------------------------

fuel_map = {
    "CNG": 0,
    "Diesel": 1,
    "LPG": 2,
    "Petrol": 3
}

seller_map = {
    "Dealer": 0,
    "Individual": 1,
    "Trustmark Dealer": 2
}

transmission_map = {
    "Automatic": 0,
    "Manual": 1
}

owner_map = {
    "First Owner": 0,
    "Fourth & Above Owner": 1,
    "Second Owner": 2,
    "Test Drive Car": 3,
    "Third Owner": 4
}

# Predict

if st.button("Predict"):

    data = np.array([[
        year,
        km_driven,
        fuel_map[fuel],
        seller_map[seller],
        transmission_map[transmission],
        owner_map[owner],
        mileage,
        engine,
        max_power,
        seats
    ]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("✅ Predicted Category: Expensive Car")
    else:
        st.warning("💰 Predicted Category: Cheap Car")
