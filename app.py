import streamlit as st
import joblib
import pandas as pd

# Load Model
model = joblib.load("model.pkl")

st.set_page_config(
    page_title="Profit Prediction",
    layout="centered"
)

st.title("📈 Superstore Profit Prediction")

st.write("Enter the product details")

sales = st.number_input(
    "Sales",
    min_value=0.0,
    value=500.0
)

quantity = st.number_input(
    "Quantity",
    min_value=1,
    value=2
)

discount = st.number_input(
    "Discount",
    min_value=0.0,
    max_value=1.0,
    value=0.10,
    step=0.01
)

ship_mode = st.selectbox(
    "Ship Mode",
    ["First Class", "Second Class", "Standard Class", "Same Day"]
)

segment = st.selectbox(
    "Segment",
    ["Consumer", "Corporate", "Home Office"]
)

category = st.selectbox(
    "Category",
    ["Furniture", "Office Supplies", "Technology"]
)

# -------- Label Encoding --------

ship_dict = {
    "First Class":0,
    "Same Day":1,
    "Second Class":2,
    "Standard Class":3
}

segment_dict = {
    "Consumer":0,
    "Corporate":1,
    "Home Office":2
}

category_dict = {
    "Furniture":0,
    "Office Supplies":1,
    "Technology":2
}

if st.button("Predict Profit"):

    data = pd.DataFrame([[
    ship_dict[ship_mode],
    segment_dict[segment],
    category_dict[category],
    sales,
    quantity,
    discount
]],
columns=[
    "Ship Mode",
    "Segment",
    "Category",
    "Sales",
    "Quantity",
    "Discount"
])

    prediction = model.predict(data)

    st.success(f"Predicted Profit : ₹ {prediction[0]:.2f}")