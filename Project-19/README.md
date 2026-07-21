# 🥛 Milk Quality Prediction using Artificial Neural Network (ANN)

An Artificial Neural Network (ANN) based Machine Learning project that predicts the quality grade of milk using its physical and chemical properties. The model is deployed using Streamlit with a simple and interactive user interface.

---

## 📌 Project Overview

This project uses an Artificial Neural Network (ANN) to classify milk quality into different grades based on various milk characteristics such as pH, temperature, taste, odor, fat content, turbidity, and colour.

The application provides real-time predictions through an easy-to-use Streamlit web interface.

---

## 🚀 Features

- Predict milk quality instantly
- ANN-based classification model
- Interactive Streamlit interface
- User-friendly input form
- Real-time prediction results
- Clean and lightweight application

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Joblib

---

## 📂 Project Structure

```
Milk_Quality_ANN/
│
├── app.py
├── milknew.csv
├── milk_quality_ann.keras
├── scaler.pkl
├── requirements.txt
├── README.md
```

---

## 📊 Input Features

- pH
- Temperature
- Taste
- Odor
- Fat
- Turbidity
- Colour

---

## 🎯 Output

The model predicts the milk quality grade:

- 🟢 High Quality
- 🟡 Medium Quality
- 🔴 Low Quality

---

## 🧠 ANN Architecture

- Input Layer
- Hidden Layer (ReLU)
- Hidden Layer (ReLU)
- Hidden Layer (ReLU)
- Output Layer (Softmax)

Optimizer:
- Adam

Loss Function:
- Sparse Categorical Crossentropy

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

## 📈 Dataset

The dataset contains different milk quality parameters including pH, temperature, taste, odor, fat, turbidity, colour, and the corresponding quality grade.

---

## 💻 Future Improvements

- Better UI Design
- Prediction Confidence Score
- Charts and Visualizations
- Batch Prediction
- Model Performance Dashboard

---

## 👨‍💻 Author

Khush Arora
