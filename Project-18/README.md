# 🍷 Wine Quality Prediction using Artificial Neural Network (ANN)

A Machine Learning web application built with TensorFlow/Keras and Streamlit to predict wine quality based on its chemical properties.

## 📌 Project Overview

This project uses an Artificial Neural Network (ANN) to predict the quality of red wine using 11 physicochemical features. The model is trained on the Wine Quality dataset and deployed with a simple and interactive Streamlit interface.

---

## 🚀 Features

- Predict wine quality using ANN
- Interactive Streamlit GUI
- Fast real-time predictions
- Clean and user-friendly interface
- TensorFlow/Keras based model

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- Scikit-learn

---

## 📂 Project Structure

```
Wine_Quality_ANN/
│
├── app.py
├── wine_quality_ann.keras
├── winequality-red.csv
├── requirements.txt
├── README.md
```

---

## 📊 Input Features

- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

---

## 🎯 Output

The application predicts the wine quality score and displays the quality category:

- 🍷 Poor Quality
- 🍷 Average Quality
- 🍷 Excellent Quality

---

## ▶️ Installation

Clone the repository

```bash
git clone <repository-url>
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

## 📈 Model Information

- Model: Artificial Neural Network (ANN)
- Hidden Layers: 3
- Activation Function: ReLU
- Output Activation: Linear
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)

---

## 📄 Dataset

Wine Quality Dataset containing physicochemical properties of red wine samples.

---

## 👨‍💻 Author

Khush Arora