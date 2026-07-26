# 🔐 VisionLock AI

> **AI Powered Family Recognition Smart Lock using Deep Learning & CNN**

VisionLock AI is a Deep Learning-based smart family recognition system built using **Convolutional Neural Networks (CNN)**. The project is designed to recognize authorized family members and distinguish them from unknown individuals, serving as the software foundation for an AI-powered smart door lock system.

Instead of relying on traditional authentication methods like keys or passwords, VisionLock AI explores how **Computer Vision** and **Deep Learning** can be used to build intelligent home security solutions.

---

# 🚀 Features

- 🧠 CNN-based Deep Learning Model
- 👨‍👩‍👦 Family Member Recognition
- 🚫 Unknown Person Detection
- 📊 Confidence Score Prediction
- 🔓 Smart Access Granted
- ❌ Smart Access Denied
- 🌐 Interactive Streamlit Web Application
- ⚡ Real-Time Image Classification

---

# 🛠️ Tech Stack

- Python
- TensorFlow
- Keras
- Convolutional Neural Networks (CNN)
- NumPy
- Pillow
- Streamlit

---

# 📂 Project Structure

```text
VisionLock_AI/
│
├── Dataset/
│   ├── training/
│   ├── validation/
│   └── testing/
│
├── visionlock_model.keras
├── app.py
├── requirements.txt
└── README.md
```

---

# 🧠 Deep Learning Pipeline

```
Image Input
      │
      ▼
Image Preprocessing
      │
      ▼
CNN Model
      │
      ▼
Feature Extraction
      │
      ▼
Multi-Class Classification
      │
      ▼
Confidence Score
      │
      ▼
Access Granted / Access Denied
```

---

# 👨‍👩‍👦 Classes Used

The model is trained on four classes:

- 👤 Khush
- 👩 Mummy
- 👨 Papa
- 🚫 Unknown

---

# 💻 Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/VisionLock-AI.git
```

### Navigate to the Project

```bash
cd VisionLock-AI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

# 📊 Sample Output

### Authorized User

```
Detected Person : Papa

Confidence : 99.62%

Status

✅ Access Granted
```

### Unknown Visitor

```
Detected Person : Unknown

Confidence : 98.41%

Status

❌ Access Denied
```

---

# 🎯 Project Vision

VisionLock AI is more than just an image classification project.

The complete Deep Learning pipeline has been successfully developed—from image preprocessing and CNN model training to prediction and deployment through a Streamlit application.

The next phase of this project focuses on integrating the trained model with:

- 📷 Camera Module
- 🔐 Electronic Door Lock
- 🤖 IoT Hardware

This integration will transform VisionLock AI into a real-world **AI-powered Family Recognition Smart Lock**, capable of granting access only to authorized family members while denying unknown visitors.

---

# 🌍 Real-World Applications

- Smart Home Security
- AI-Based Door Lock Systems
- Office Access Control
- Visitor Authentication
- Educational Deep Learning Projects
- Computer Vision Research

---

# 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Deep Learning Fundamentals
- Convolutional Neural Networks (CNN)
- Image Preprocessing
- Multi-Class Image Classification
- TensorFlow & Keras
- Model Deployment using Streamlit
- Building AI Applications with Real-World Use Cases

---

# 🚀 Future Enhancements

- 📷 Live Camera Recognition
- 😊 Face Detection Before Classification
- 🔔 Mobile Notifications
- ☁ Cloud Deployment
- 🔐 IoT Smart Lock Integration
- 📱 Mobile Application
- ⚡ Real-Time Video Inference

---

# 📸 Project Preview

> Add screenshots here

### 🏠 Home Screen

![Home](screenshots/home.png)

---

### 📤 Upload Image

![Upload](screenshots/upload.png)

---

### ✅ Access Granted

![Granted](screenshots/granted.png)

---

### ❌ Access Denied

![Denied](screenshots/denied.png)

---

# 🤝 Contributing

Contributions, suggestions, and improvements are always welcome.

Feel free to fork the repository and submit a pull request.

---

# 👨‍💻 Author

**Khush Arora**

If you found this project interesting, consider giving it a ⭐ on GitHub.

---

## ⭐ Star this repository if you like the project!