import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

st.set_page_config(page_title="Smart Customer Segmentation", layout="wide")

st.title("Smart Customer Segmentation System using K-Means Clustering")

MODEL_PATH = "kmeans_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "gender_encoder.pkl"
DATA_PATH = "Mall_Customers.csv"

missing_files = [f for f in [MODEL_PATH, SCALER_PATH, ENCODER_PATH, DATA_PATH] if not os.path.exists(f)]

if missing_files:
    st.error(f"Missing required files: {', '.join(missing_files)}. Please ensure all model files and dataset are present in the project folder.")
    st.stop()

kmeans_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
gender_encoder = joblib.load(ENCODER_PATH)
df = pd.read_csv(DATA_PATH)

cluster_labels = list(range(kmeans_model.cluster_centers_.shape[0]))

cluster_names_pool = {
    0: "Budget Customer",
    1: "Premium Customer",
    2: "Regular Customer",
    3: "Luxury Customer",
    4: "High Potential Customer",
}

cluster_descriptions = {
    "Budget Customer": "Customers with lower income and spending, focused on essential purchases.",
    "Premium Customer": "Customers with high income and high spending, valuing quality and exclusivity.",
    "Regular Customer": "Customers with moderate income and balanced spending habits.",
    "Luxury Customer": "Customers with very high income who spend generously on premium products.",
    "High Potential Customer": "Customers with promising spending behavior who could be nurtured into loyal buyers.",
}

cluster_recommendations = {
    "Budget Customer": "Offer discounts, value bundles, and budget-friendly promotions.",
    "Premium Customer": "Provide exclusive offers, loyalty rewards, and premium product recommendations.",
    "Regular Customer": "Engage with seasonal offers and personalized recommendations.",
    "Luxury Customer": "Target with high-end products, VIP services, and early access to new collections.",
    "High Potential Customer": "Nurture with targeted campaigns and incentives to increase engagement.",
}

cluster_map = {c: cluster_names_pool.get(c, f"Cluster {c} Customer") for c in cluster_labels}

st.sidebar.header("Customer Input Features")
gender_input = st.sidebar.selectbox("Gender", ["Male", "Female"])
age_input = st.sidebar.slider("Age", 18, 70, 30)
income_input = st.sidebar.slider("Annual Income (k$)", 10, 150, 50)
spending_input = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

encoded_gender = gender_encoder.transform([gender_input])[0]

input_df = pd.DataFrame({
    "Age": [age_input],
    "Annual Income (k$)": [income_input],
    "Spending Score (1-100)": [spending_input],
})

scaled_input = scaler.transform(input_df)
predicted_cluster = int(kmeans_model.predict(scaled_input)[0])
customer_type = cluster_map.get(predicted_cluster, f"Cluster {predicted_cluster} Customer")
description = cluster_descriptions.get(customer_type, "No description available.")
recommendation = cluster_recommendations.get(customer_type, "No recommendation available.")

st.subheader("Prediction Result")
st.success(f"Customer Cluster: {predicted_cluster}")
st.info(f"Customer Type: {customer_type}\n\nDescription: {description}\n\nMarketing Recommendation: {recommendation}")

st.subheader("Dataset Preview")
st.dataframe(df.head(10))

st.subheader("Customer Analytics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", len(df))
col2.metric("Average Age", round(df["Age"].mean(), 2))
col3.metric("Average Income (k$)", round(df["Annual Income (k$)"].mean(), 2))
col4.metric("Average Spending Score", round(df["Spending Score (1-100)"].mean(), 2))

df_scaled_all = scaler.transform(df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])
df["Cluster"] = kmeans_model.predict(df_scaled_all)
df["Customer Type"] = df["Cluster"].map(cluster_map)

st.subheader("Cluster Visualization")
col_a, col_b = st.columns(2)

with col_a:
    fig_scatter = px.scatter(
        df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color=df["Cluster"].astype(str),
        hover_data=["Customer Type"],
        title="Customer Segments by Income and Spending Score",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_b:
    fig_hist = px.histogram(df, x="Age", nbins=20, title="Age Distribution")
    st.plotly_chart(fig_hist, use_container_width=True)

fig_pie = px.pie(df, names="Gender", title="Gender Distribution")
st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")
st.markdown("Developed using Streamlit + Scikit-Learn + KMeans")
