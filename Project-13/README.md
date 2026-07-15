# ❤️ Heart Disease Prediction using Decision Tree Classifier

This project predicts the presence of heart disease using the Decision Tree Classification algorithm. A Streamlit web application is used to collect patient information and display the prediction result.

---

## 📌 Project Overview

This machine learning project is built using the Heart Disease dataset. The model is trained using the Decision Tree Classifier algorithm and deployed with Streamlit to provide an interactive prediction interface.

---

## 🚀 Features

- Predicts whether a patient has heart disease.
- Interactive Streamlit web application.
- Trained using Decision Tree Classifier.
- Model saved using Joblib.
- Fast and user-friendly prediction system.

---

## 📂 Project Structure

```
Heart_Disease_Prediction/
│
├── heart.csv
├── ds day13(Decision tree classifier).ipynb
├── decision_tree_model.pkl
├── app.py
├── README.md
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

---

## 📊 Input Features

- Age
- Sex
- Chest Pain Type (cp)
- Resting Blood Pressure (trestbps)
- Cholesterol (chol)
- Fasting Blood Sugar (fbs)
- Resting ECG (restecg)
- Maximum Heart Rate (thalach)
- Exercise Induced Angina (exang)
- Oldpeak
- Slope
- Number of Major Vessels (ca)
- Thal

---

## 🎯 Output

The model predicts one of the following:

- ✅ No Heart Disease Detected
- ⚠️ Heart Disease Detected

---

## 📈 Model Performance

- Algorithm: Decision Tree Classifier
- Train-Test Split: 80:20
- Training Accuracy: **100%**
- Testing Accuracy: **98.54%**

---

## ▶️ How to Run

### Install Required Libraries

```bash
pip install pandas numpy scikit-learn streamlit joblib
```

### Run the Application

```bash
streamlit run app.py
```

---

## 👨‍💻 Author

**Khush Arora**

Data Science Summer Internship 2026