import streamlit as st
import pickle
import numpy as np

# Load Model
model = pickle.load(open("Car_Price_Prediction.pkl", "rb"))

st.title("🚗 Car Price Prediction App")
st.write("Enter Car Details to Predict Price")

# Inputs
year = st.number_input("Year", min_value=1990, max_value=2026, value=2018)

km_driven = st.number_input("KM Driven", min_value=0, value=50000)

fuel = st.selectbox("Fuel", ["CNG", "Diesel", "LPG", "Petrol"])

seller_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual", "Trustmark Dealer"]
)

transmission = st.selectbox(
    "Transmission",
    ["Automatic", "Manual"]
)

owner = st.selectbox(
    "Owner",
    [
        "First Owner",
        "Fourth & Above Owner",
        "Second Owner",
        "Test Drive Car",
        "Third Owner"
    ]
)

mileage = st.number_input("Mileage (km/ltr)", value=20.0)

engine = st.number_input("Engine (CC)", value=1200.0)

max_power = st.number_input("Max Power", value=80.0)

seats = st.number_input("Seats", value=5.0)

# Label Encoding Mapping
fuel_map = {
    "CNG":0,
    "Diesel":1,
    "LPG":2,
    "Petrol":3
}

seller_map = {
    "Dealer":0,
    "Individual":1,
    "Trustmark Dealer":2
}

transmission_map = {
    "Automatic":0,
    "Manual":1
}

owner_map = {
    "First Owner":0,
    "Fourth & Above Owner":1,
    "Second Owner":2,
    "Test Drive Car":3,
    "Third Owner":4
}

if st.button("Predict Price"):

    data = np.array([[

        year,
        km_driven,
        fuel_map[fuel],
        seller_map[seller_type],
        transmission_map[transmission],
        owner_map[owner],
        mileage,
        engine,
        max_power,
        seats

    ]])

    prediction = model.predict(data)

    st.success(f"Estimated Car Price : ₹ {prediction[0]:,.2f}")