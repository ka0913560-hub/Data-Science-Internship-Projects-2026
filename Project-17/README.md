# 🛍️ Smart Customer Segmentation using K-Means Clustering

A beginner-friendly Machine Learning project that groups mall customers into meaningful segments using the **K-Means Clustering** algorithm, and deploys the trained model as an interactive **Streamlit web application**.

---

## 📌 2. Project Description

Every mall or retail store has hundreds or thousands of customers, and each customer behaves differently. Some customers spend a lot of money, some spend very little, some earn a high income but spend cautiously, and some earn less but spend a lot.

If a business tries to market the same offer to every customer, it usually fails because customers are not all the same. This is where **Customer Segmentation** becomes useful.

This project uses **unsupervised machine learning (K-Means Clustering)** to automatically group customers into different segments based on their **Age**, **Annual Income**, and **Spending Score**. Once customers are grouped, a business can design **targeted marketing strategies** for each group instead of using a "one-size-fits-all" approach.

The project also includes a **Streamlit web application** where a user can input a customer's details and instantly find out which segment that customer belongs to.

---

## 🎯 3. Objective

The main objectives of this project are:

- To understand and apply the **K-Means Clustering algorithm** on real-world data.
- To segment mall customers into distinct groups based on their income and spending behavior.
- To help businesses identify **high-value customers** and **low-engagement customers**.
- To build a simple and interactive **Streamlit dashboard** for real-time cluster prediction.
- To learn the complete ML lifecycle: data exploration → preprocessing → model training → evaluation → deployment.

---

## 🧠 4. About K-Means Clustering

### 🔹 What is K-Means?

K-Means is a **clustering algorithm** used to divide a dataset into a fixed number (`K`) of groups, called **clusters**, based on similarity. Data points that are similar to each other are placed in the same cluster, and data points that are different are placed in different clusters.

### 🔹 Why is it called an Unsupervised Learning Algorithm?

K-Means is called **unsupervised** because it does **not use labeled data**. There is no "correct answer" or "target column" given to the model beforehand. The algorithm looks at the data on its own and finds hidden patterns or natural groupings, without any human telling it what the groups should be.

### 🔹 Why do we use K-Means?

- It is simple, fast, and easy to understand.
- It works very well when we want to discover natural groupings in data.
- It does not require labeled/tagged data, which is expensive and time-consuming to create.
- It scales well to large datasets.

### 🔹 Real-World Applications of K-Means

| Application | Description |
|---|---|
| 🛒 Customer Segmentation | Grouping customers by purchasing behavior |
| 📚 Document Clustering | Grouping similar articles or news |
| 🏥 Medical Diagnosis | Grouping patients with similar symptoms |
| 🌍 Geolocation Analysis | Grouping delivery zones or store locations |
| 🖼️ Image Compression | Reducing colors in an image |
| 🏦 Fraud Detection | Identifying unusual transaction patterns |

### 🔹 Advantages of K-Means

- ✅ Easy to understand and implement
- ✅ Fast and computationally efficient
- ✅ Works well with large datasets
- ✅ Produces easily interpretable clusters

### 🔹 Limitations of K-Means

- ❌ We must specify the number of clusters (`K`) in advance
- ❌ Sensitive to outliers
- ❌ Assumes clusters are spherical/circular in shape
- ❌ Results can change depending on initial centroid positions
- ❌ Not effective on categorical data without proper encoding

---

## 📊 5. Why this Dataset?

The **Mall Customer Dataset** is one of the most popular datasets used for practicing customer segmentation because:

- It contains simple, real-world-like customer attributes: **Age**, **Gender**, **Annual Income**, and **Spending Score**.
- The dataset is small, clean, and beginner-friendly, making it easy to visualize clusters.
- It naturally contains groups of customers with different income and spending patterns, which is perfect for demonstrating how K-Means works.
- Businesses genuinely use similar data (income + spending behavior) to run marketing campaigns, so this dataset closely reflects a real business use case.

---

## 📁 6. Dataset Information

The dataset `Mall_Customers.csv` contains the following columns:

| Column Name | Description |
|---|---|
| **CustomerID** | A unique ID assigned to every customer. It has no meaningful pattern and is only used for identification. |
| **Gender** | The gender of the customer (Male/Female). Useful for demographic analysis. |
| **Age** | The age of the customer in years. Helps understand age-based spending trends. |
| **Annual Income (k$)** | The yearly income of the customer in thousands of dollars. Indicates the customer's purchasing power. |
| **Spending Score (1-100)** | A score assigned by the mall based on customer behavior and spending nature. A higher score means the customer spends more. |

> ⚠️ **Note:** `CustomerID` is **not used for training** the model because it is just a unique identifier and carries no useful pattern or relationship with customer behavior.

---

## 🔄 7. Project Workflow

### **Step 1: Import Libraries**

We import all the required Python libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn`. These libraries help us load data, visualize it, and build the machine learning model.

### **Step 2: Load Dataset**

The `Mall_Customers.csv` file is loaded into a pandas DataFrame using `pd.read_csv()`. This allows us to view and manipulate the data easily in table format.

### **Step 3: Data Exploration**

We explore the dataset using functions like `.head()`, `.info()`, `.describe()`, and `.shape` to understand:
- Number of rows and columns
- Data types of each column
- Statistical summary (mean, min, max, etc.)

### **Step 4: Check Missing Values**

We use `.isnull().sum()` to check whether any column has missing (NaN) values. Missing values can cause errors or biased results in the model, so it's important to identify and handle them.

### **Step 5: Check Duplicate Values**

We use `.duplicated().sum()` to check if there are any repeated rows in the dataset. Duplicate rows can affect the accuracy of the clustering process, so they must be identified and removed if found.

### **Step 6: Visualizations**

We create graphs like histograms, scatter plots, and count plots to understand:
- The distribution of Age, Income, and Spending Score
- The relationship between Income and Spending Score
- The gender distribution of customers

These visualizations help us understand patterns **before** applying any algorithm.

### **Step 7: Feature Selection**

We select only these three features for clustering:

- **Age**
- **Annual Income (k$)**
- **Spending Score (1-100)**

**Why only these features?**
- These three features directly represent a customer's **spending behavior and profile**.
- `CustomerID` has no analytical value and is excluded.
- `Gender` is a categorical column, and adding it without strong justification can distort distance-based calculations used in K-Means. Hence, the model focuses on numerical, behavior-driven features that best describe customer segments.

### **Step 8: Standard Scaling**

We use `StandardScaler` from scikit-learn to scale the selected features.

**Why is scaling required?**
K-Means uses **distance calculations** (like Euclidean distance) to group data points. If one feature (like Annual Income, ranging from 10–150) has a much larger scale than another feature (like Spending Score, ranging from 1–100), the algorithm will give more importance to the larger-scale feature unfairly. Scaling brings all features to a similar range (mean = 0, standard deviation = 1), ensuring every feature contributes equally to the clustering process.

### **Step 9: Elbow Method**

**What is WCSS?**
WCSS stands for **Within-Cluster Sum of Squares**. It measures the sum of squared distances between each data point and the centroid of its assigned cluster.

**Why do we calculate WCSS?**
WCSS tells us how tightly the data points are grouped within a cluster. A lower WCSS means the points are closer to their centroid, indicating better/tighter clusters.

**Why is the Elbow Method used?**
Since K-Means requires us to define the number of clusters (`K`) beforehand, the Elbow Method helps us choose the **best value of K**. We plot WCSS against different values of K (e.g., 1 to 10), and the graph typically looks like a bent arm.

**How is the best number of clusters selected?**
The point where the WCSS curve starts to flatten out (like an elbow bend) is chosen as the optimal K. Beyond this point, increasing K does not significantly reduce WCSS, meaning adding more clusters gives diminishing returns.

### **Step 10: Train K-Means Model**

**What does `KMeans(n_clusters=5)` mean?**
This creates a K-Means model that will divide the data into **5 clusters**. The number `5` is chosen based on the Elbow Method result.

**What is a centroid?**
A **centroid** is the center point of a cluster. It is the average position of all data points that belong to that cluster.

**How are initial centroids selected?**
Initially, K-Means picks `K` random data points (or uses smarter techniques like `k-means++`) as starting centroids. These are then refined over several iterations to reach their optimal position.

### **Step 11: How K-Means Creates Clusters**

K-Means works in simple repeated steps:

1. **Random Centroids** — The algorithm randomly picks `K` points from the data as the initial centroids (center points of clusters).
2. **Distance Calculation** — For every data point, the algorithm calculates the distance between that point and every centroid.
3. **Assign Nearest Centroid** — Each data point is assigned to the cluster whose centroid is closest to it.
4. **Update Centroid** — After all points are assigned, the algorithm recalculates the centroid of each cluster by taking the average of all points in that cluster.
5. **Repeat Until Convergence** — Steps 2 to 4 are repeated again and again. Each time, points may switch to a closer cluster, and centroids move slightly. This continues until the centroids stop moving (or move very little), meaning the clusters have stabilized.

In very simple words: *K-Means keeps rearranging customers into groups and moving the "center" of each group until everyone is in the group that fits them best.*

### **Step 12: Prediction**

**How does `model.predict()` work?**
Once the model is trained, `model.predict()` takes a new data point (a new customer's Age, Income, and Spending Score) and calculates its distance from all existing centroids. It then assigns the new customer to the cluster whose centroid is **nearest** to that customer.

This is exactly how the Streamlit app predicts the segment for a new customer entered by the user.

### **Step 13: Silhouette Score**

**Formula:**

$$
S = \frac{b - a}{\max(a, b)}
$$

Where:
- `a` = average distance between a point and all other points in the **same cluster**
- `b` = average distance between a point and all points in the **nearest different cluster**

**Meaning:** The Silhouette Score measures how well a data point fits within its own cluster compared to other clusters.

**Range:** The score ranges from **-1 to +1**.

**Why higher is better:**
- A score close to **+1** means the point is well-matched to its own cluster and poorly matched to neighboring clusters (good clustering).
- A score close to **0** means the point is on the border between two clusters.
- A score close to **-1** means the point may have been assigned to the wrong cluster.

### **Step 14: WCSS (Inertia)**

**What does inertia mean?**
Inertia (also called WCSS) is the sum of squared distances between each data point and its assigned cluster's centroid. It tells us how compact the clusters are. Lower inertia means the data points are closer to their centroids, indicating well-formed clusters. In scikit-learn, this value can be accessed using `model.inertia_`.

### **Step 15: Saving Model**

**Why is `joblib.dump()` used?**
`joblib.dump()` is used to save a trained model, scaler, or encoder into a file (`.pkl` format) so that it can be reused later **without retraining** it every time.

**Why are models saved before deploying in Streamlit?**
Training a machine learning model can take time and computing resources. Once a model is trained and finalized, we save it using `joblib`. The Streamlit app then simply **loads** this saved model (`kmeans_model.pkl`), the saved `scaler.pkl`, and `gender_encoder.pkl` to make instant predictions, without going through the training process again. This makes the application fast, efficient, and production-ready.

---

## 💻 8. Streamlit Application

The Streamlit application (`app.py`) provides an interactive interface for predicting a customer's segment in real time.

### How the Application Works

1. The user enters details in the sidebar:
   - **Gender**
   - **Age**
   - **Annual Income (k$)**
   - **Spending Score (1-100)**

2. **Encoding:** The entered Gender value is converted into a numeric form using the saved `gender_encoder.pkl`.

3. **Feature Preparation:** A small dataframe is created using only the three features the model was trained on — **Age**, **Annual Income (k$)**, and **Spending Score (1-100)**.

4. **Scaling:** This dataframe is transformed using the saved `scaler.pkl`, so the new customer's data is on the same scale as the training data.

5. **Prediction:** The scaled data is passed to `kmeans_model.predict()`, which calculates the distance between the new customer's data point and all cluster centroids, and returns the **nearest cluster number**.

6. **Result Display:** The predicted cluster number is mapped to a meaningful customer type (e.g., "Premium Customer"), along with a short description and marketing recommendation, shown using `st.success()` and `st.info()`.

The app also displays the dataset preview, key analytics (total customers, average age, average income, average spending score), and interactive Plotly charts (scatter plot, histogram, and pie chart) for deeper insights.

---

## 🧩 9. Customer Cluster Types

After training, K-Means only returns numeric cluster labels like `0`, `1`, `2`, `3`, `4`. These numbers by themselves have **no business meaning**. To make the results understandable for non-technical users, we manually study the characteristics (average income and spending score) of each cluster and assign meaningful names, such as:

| Cluster Name | Typical Characteristics |
|---|---|
| 💰 **Budget Customer** | Low income, low spending score |
| 🧾 **Regular Customer** | Moderate income, moderate spending score |
| 💎 **Premium Customer** | High income, high spending score |
| 👑 **Luxury Customer** | Very high income, very high spending score |
| 🌱 **High Potential Customer** | Low/medium income but high spending score (could be nurtured further) |

> ⚠️ **Important:** These names are **manually assigned by the developer** after analyzing each cluster's characteristics. K-Means itself has no concept of "Premium" or "Budget" — it only groups similar data points and returns a **cluster number**. The meaningful labels are added afterward based on human interpretation of the data.

---

## 🛠️ 10. Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 **Python** | Core programming language |
| 🐼 **Pandas** | Data loading and manipulation |
| 🔢 **NumPy** | Numerical operations |
| 📊 **Matplotlib** | Data visualization |
| 🎨 **Seaborn** | Statistical data visualization |
| 🤖 **Scikit-Learn** | Building the K-Means model, scaling, and evaluation |
| 🌐 **Streamlit** | Building the interactive web application |
| 💾 **Joblib** | Saving and loading trained model, scaler, and encoder |

---

## 📂 11. Project Folder Structure

```
Day 15(Mall Customer KMeans)/
│
├── Mall_Customers.csv          # Dataset used for training and analysis
├── Mall_Customer_KMean.ipynb   # Jupyter Notebook with EDA & model training
├── kmeans_model.pkl            # Saved trained K-Means model
├── scaler.pkl                  # Saved StandardScaler object
├── gender_encoder.pkl          # Saved encoder for Gender column
├── app.py                      # Streamlit web application
└── README.md                   # Project documentation
```

---

## ⚙️ 12. Installation

Follow these steps to run the project on your local machine:

**1. Install the required dependencies:**

```bash
pip install streamlit pandas numpy joblib matplotlib seaborn plotly scikit-learn
```

**2. Run the Streamlit application:**

```bash
streamlit run app.py
```

The app will open automatically in your default web browser.

---

## 🚀 13. Future Improvements

1. Add more features like purchase frequency, product categories, and visit duration for richer segmentation.
2. Implement other clustering algorithms (DBSCAN, Hierarchical Clustering) for comparison.
3. Automate optimal cluster selection using Silhouette Score alongside the Elbow Method.
4. Add a feature to upload a custom CSV file and get bulk cluster predictions.
5. Deploy the application on cloud platforms like Streamlit Cloud, Heroku, or AWS.
6. Add authentication so businesses can securely manage their own customer data.
7. Include a recommendation engine that suggests specific products for each cluster.
8. Add downloadable PDF/Excel reports summarizing each customer segment.
9. Integrate real-time database connections (SQL/NoSQL) instead of a static CSV file.
10. Add unit tests and CI/CD pipeline for better code reliability and deployment automation.

---

## 📝 14. Conclusion

This project provided hands-on experience with the complete machine learning workflow — starting from data exploration, cleaning, and visualization, to feature selection, scaling, model training, evaluation, and finally deployment. 

Through this project, we learned how **K-Means Clustering**, an unsupervised learning algorithm, can uncover hidden patterns in customer data without needing labeled outcomes. We also learned the importance of **feature scaling**, how to choose the optimal number of clusters using the **Elbow Method**, and how to evaluate cluster quality using the **Silhouette Score**.

Finally, by saving the trained model using `joblib` and building a **Streamlit web application**, we understood how machine learning models are deployed in the real world to deliver instant, practical value to businesses — turning raw customer data into actionable marketing insights.

---

## ❓ 15. Interview / Viva Questions

**1. What is K-Means Clustering?**
K-Means is an unsupervised machine learning algorithm that groups data into `K` clusters based on similarity, where each data point belongs to the cluster with the nearest centroid.

**2. Why is K-Means called an unsupervised algorithm?**
Because it does not use labeled data. It finds patterns and groups in the data on its own without being told the "correct" output.

**3. What is a centroid in K-Means?**
A centroid is the center point of a cluster, calculated as the average of all data points belonging to that cluster.

**4. How does K-Means algorithm work?**
It randomly initializes centroids, assigns each data point to the nearest centroid, recalculates centroids based on the assigned points, and repeats this process until the centroids no longer change significantly (convergence).

**5. How do you choose the value of K in K-Means?**
The value of K is chosen using techniques like the **Elbow Method** or **Silhouette Score**, which help identify the number of clusters that best represents the data's natural groupings.

**6. What is the Elbow Method?**
It is a technique where WCSS is plotted against different values of K. The point where the curve bends like an elbow indicates the optimal number of clusters.

**7. What is WCSS?**
WCSS (Within-Cluster Sum of Squares) is the sum of squared distances between each data point and its cluster's centroid. It measures how compact the clusters are.

**8. What is inertia in K-Means?**
Inertia is another name for WCSS. It represents how tightly the data points are clustered around their centroids; a lower value indicates better clustering.

**9. What is the Silhouette Score?**
It is a metric used to evaluate the quality of clusters by measuring how similar a data point is to its own cluster compared to other clusters. It ranges from -1 to +1.

**10. Why is a higher Silhouette Score considered better?**
A higher score means data points are well-matched to their own cluster and clearly separated from other clusters, indicating better-defined clusters.

**11. Why is feature scaling important before applying K-Means?**
K-Means relies on distance calculations. If features have different scales (e.g., income vs. spending score), the feature with a larger range will dominate the distance calculation, leading to biased clusters. Scaling ensures all features contribute equally.

**12. Which scaling technique is used in this project, and why?**
`StandardScaler` is used because it transforms data to have a mean of 0 and standard deviation of 1, making all features comparable regardless of their original scale.

**13. Why was CustomerID excluded from the model?**
Because it is a unique identifier with no meaningful pattern or relationship with customer behavior, so it adds no value to clustering.

**14. Why were only Age, Annual Income, and Spending Score selected as features?**
These three features directly represent the customer's demographic and spending behavior, making them most relevant for segmentation, while other columns like CustomerID hold no analytical value.

**15. How does the model predict the cluster for a new customer?**
The new customer's data is scaled using the saved scaler, and then `model.predict()` calculates the distance between this data point and each cluster's centroid, assigning it to the nearest cluster.

**16. Why do we save the model using joblib instead of retraining it every time?**
Training a model takes time and computational resources. Saving it with `joblib.dump()` allows us to reuse the trained model instantly for predictions without repeating the training process.

**17. What is the purpose of the gender_encoder.pkl file?**
It converts the categorical Gender values (Male/Female) into numeric form so that they can be used consistently with the trained model, if required.

**18. What is the difference between supervised and unsupervised learning?**
Supervised learning uses labeled data with known outputs to train a model, while unsupervised learning works with unlabeled data to discover hidden patterns or groupings, like clustering.

**19. Why might K-Means give different results on different runs?**
Because the initial centroids are chosen randomly, different runs might converge to slightly different cluster arrangements unless a fixed `random_state` is used.

**20. What are some limitations of K-Means clustering?**
K-Means requires specifying the number of clusters beforehand, is sensitive to outliers, assumes clusters are spherical in shape, and may produce different results based on initial centroid selection.

**21. What is customer segmentation, and why is it useful for businesses?**
Customer segmentation is the process of dividing customers into groups based on shared characteristics like income and spending behavior. It helps businesses design targeted marketing strategies, improve customer satisfaction, and increase revenue.

**22. Why are cluster names like "Premium Customer" or "Budget Customer" manually assigned?**
Because K-Means only outputs numeric cluster labels (0, 1, 2, etc.) with no inherent meaning. Developers analyze each cluster's average characteristics and assign meaningful business-friendly names accordingly.

---

⭐ **Developed using Streamlit + Scikit-Learn + KMeans**
