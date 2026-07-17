import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("Data for repository.csv")

# Encode Genre
le = LabelEncoder()
df["Genre_Code"] = le.fit_transform(df["Genre"])

st.title("🎬 Movie Recommendation System")

# Genre Dropdown
genre_name = st.selectbox("Select Genre", le.classes_)

# Selected Genre ka Code
genre_code = le.transform([genre_name])[0]

if st.button("Recommend Movies"):

    movies = df[df["Genre_Code"] == genre_code]["Movie_Name"].tolist()

    st.subheader(f"Top 10 {genre_name} Movies")

    for movie in movies[:10]:
        st.write("🎥", movie)

    if len(movies) > 10:
        with st.expander("📂 Show More Movies"):
            for movie in movies[10:]:
                st.write("🎥", movie)
