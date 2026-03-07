# 🚀 Customer Churn Prediction MLOps System

## 📌 Overview

This project demonstrates an **end-to-end Machine Learning and MLOps pipeline for Customer Churn Prediction**.

The workflow starts with **model experimentation in Jupyter Notebook** and transitions into a **production-ready ML system** using modern MLOps practices including API deployment, containerization, and CI/CD automation.

Customer churn prediction helps companies identify customers who are likely to leave their service, allowing businesses to **take proactive actions to retain customers and reduce revenue loss**.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 🏗️ Project Architecture

Customer-Churn-MLOps-System

├── notebook/           # ML experimentation and model development

├── src/                # Training and pipeline scripts

├── model/              # Saved trained model

├── app.py              # FastAPI application for model inference

├── Dockerfile          # Containerization configuration

├── requirements.txt    # Python dependencies

├── .github/workflows   # CI/CD pipeline using GitHub Actions

├── .gitignore

└── README.md

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 🔬 Machine Learning Experimentation

The initial experimentation and model development were performed in **Jupyter Notebook**.

Notebook tasks include:

• Data exploration and analysis
• Data preprocessing
• Feature engineering
• Model training and evaluation
• Handling class imbalance
• Model performance comparison

After experimentation, the final model was exported and integrated into the **production pipeline**.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ⚙️ Machine Learning Pipeline

The production pipeline includes:

1️⃣ Data preprocessing

* Cleaning and preparing customer data
* Encoding categorical features

2️⃣ Model training

* Training classification model for churn prediction

3️⃣ Model evaluation

* Measuring model performance using classification metrics

4️⃣ Model serialization

* Saving trained model using Joblib

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 🌐 Model Serving API

The trained model is deployed as a **REST API using FastAPI**.

### Endpoint

POST /predict

### Example Input

{
"tenure": 12,
"monthly_charges": 65.5,
"contract_type": "Month-to-month"
}

### Example Output

{
"churn_prediction": 1
}

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 🐳 Docker Containerization

The application is containerized using Docker to ensure consistent environments across development and deployment.

Build Docker image

docker build -t churn-api .

Run Docker container

docker run -p 8000:8000 churn-api

API Documentation

http://localhost:8000/docs

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 🔁 CI/CD Pipeline

A continuous integration pipeline is implemented using **GitHub Actions**.

The pipeline automatically executes when code is pushed to the repository.

Pipeline steps include:

• Repository checkout
• Python environment setup
• Dependency installation
• Docker image build
• Container testing
• API health check
• FastAPI documentation verification

This ensures the system is **automatically tested and production ready**.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 🛠️ Technology Stack

Programming Language
Python

Data Processing
Pandas, NumPy

Machine Learning
Scikit-learn

Experimentation
Jupyter Notebook

Model Serving
FastAPI

Model Serialization
Joblib

Containerization
Docker

CI/CD
GitHub Actions

Version Control
Git

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 📊 Business Value

Customer churn prediction systems help organizations:

• Identify customers at risk of leaving
• Improve retention strategies
• Reduce revenue loss
• Enable data-driven decision making

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 🚀 Future Improvements

• MLflow experiment tracking
• Automated model retraining
• Model monitoring and drift detection
• Cloud deployment (AWS / GCP / Azure)
• Feature store integration

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 👨‍💻 Author

Akilan M
Aspiring Data Scientist | Machine Learning & MLOps Practitioner

Passionate about building end-to-end machine learning systems — from data analysis and model development to production deployment.

GitHub
https://github.com/akilanm71

