# 🌳 Decision Tree Regression - Superstore Profit Prediction

## 📌 Project Overview

This project predicts the **Profit** of Superstore orders using the **Decision Tree Regression** algorithm. The model is trained on the Superstore dataset after preprocessing and feature engineering. Hyperparameter tuning is performed using **GridSearchCV** to improve model performance. A simple **Streamlit GUI** is also developed for real-time profit prediction.

---

## 🎯 Objective

To build a machine learning model that accurately predicts the profit of a product order based on different sales-related features.

---

## 📂 Dataset

**Dataset:** Superstore Sales Dataset

### Target Variable
- Profit

### Features Used
- Ship Mode
- Segment
- Category
- Sales
- Quantity
- Discount

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit

---

## 📋 Project Workflow

1. Load Dataset
2. Data Cleaning
3. Drop Unnecessary Columns
4. Label Encoding of Categorical Features
5. Feature Selection
6. Train-Test Split
7. Train Decision Tree Regressor
8. Hyperparameter Tuning using GridSearchCV
9. Model Evaluation
10. Decision Tree Visualization
11. Actual vs Predicted Profit Comparison
12. Save Model using Joblib
13. Build Streamlit GUI

---

## ⚙️ Model Used

- DecisionTreeRegressor

Hyperparameters were optimized using **GridSearchCV**.

Example Parameters:

- criterion = squared_error
- max_depth = 3
- min_samples_split = 5
- min_samples_leaf = 2

---

## 📊 Evaluation Metrics

The model was evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

## 📈 Visualizations

- Decision Tree Plot
- Actual vs Predicted Profit Scatter Plot

---

## 💻 Streamlit GUI

The project includes a simple GUI where users can enter:

- Sales
- Quantity
- Discount
- Ship Mode
- Segment
- Category

The model predicts the expected **Profit** instantly.

---

## 📁 Project Structure

```
Decision-Tree-Regression/
│
├── app.py
├── model.pkl
├── superstore.csv
├── DecisionTreeRegression.ipynb
├── requirements.txt
└── README.md
```

---

## ▶️ Run the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## 📚 Machine Learning Concepts Covered

- Regression
- Decision Tree Regression
- Label Encoding
- Data Preprocessing
- Train-Test Split
- Hyperparameter Tuning
- GridSearchCV
- Model Evaluation
- Model Serialization (Joblib)
- Streamlit Deployment

---

## 👨‍💻 Author

**Khush Arora**

B.Tech CSE | Data Science Enthusiast

---

## ⭐ Future Improvements

- Random Forest Regression
- XGBoost Regression
- Feature Importance Visualization
- Deployment on Streamlit Cloud
- Better UI Design