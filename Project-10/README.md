# 🚗 Car Price Classification using Logistic Regression

A Machine Learning classification project that predicts whether a used car belongs to the **Cheap** or **Expensive** category using Logistic Regression. The application is developed with Streamlit and allows users to enter car details for real-time classification.

---

## 📌 Project Overview

This project uses the Cardekho dataset to classify cars into two categories:

- 💰 Cheap Car (0)
- 🚘 Expensive Car (1)

The classification is based on various car features such as year, fuel type, transmission, engine capacity, mileage, seats, and more.

---

## 🚀 Features

- Data Cleaning & Preprocessing
- Missing Value Handling
- Duplicate Removal
- Label Encoding
- Binary Classification using Logistic Regression
- Model Training & Testing
- Accuracy Evaluation
- Streamlit Web Application
- Real-Time Car Category Prediction

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle

---

## 📂 Project Structure

```
Car-Price-Classification/
│
├── app.py
├── Logistic_Regression_Model.pkl
├── cardekho.csv
├── requirements.txt
├── README.md
└── Logistic_Regression.ipynb
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/Car-Price-Classification.git
```

Move to the project folder

```bash
cd Car-Price-Classification
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📊 Dataset Features

The model uses the following features:

- Year
- KM Driven
- Fuel Type
- Seller Type
- Transmission
- Owner
- Mileage
- Engine
- Max Power
- Seats

---

## 🤖 Machine Learning Workflow

- Import Dataset
- Data Cleaning
- Handle Missing Values
- Remove Duplicate Records
- Convert Categorical Data using Label Encoding
- Create Binary Target Variable
- Train-Test Split
- Train Logistic Regression Model
- Evaluate Model
- Save Model (.pkl)
- Deploy using Streamlit

---

## 📈 Model Performance

**Algorithm:** Logistic Regression

Evaluation Metrics:

- Accuracy Score
- Confusion Matrix
- Classification Report

---

## 🚗 Prediction Output

The application predicts:

- 💰 Cheap Car
- 🚘 Expensive Car

---

## 📸 Application Preview

Add your Streamlit application screenshot here.

```
images/app.png
```

---

## 👨‍💻 Author

**Khush Arora**

B.Tech CSE Student