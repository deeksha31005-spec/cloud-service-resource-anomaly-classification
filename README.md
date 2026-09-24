# ☁️ Cloud Service Resource Anomaly Classification

A machine learning-based system for identifying unusual resource-usage observations in a simulated cloud-service environment.

## 📌 Project Overview

Cloud services generate large amounts of resource and performance data. Detecting unusual behavior from these metrics can help identify potentially anomalous periods.

This project develops an educational machine learning prototype that classifies cloud-service observations as:

- ✅ Normal
- ⚠️ Anomalous

The system uses selected cloud resource and service-performance metrics and applies machine learning for binary classification.

## 🎯 Objectives

- Analyze cloud-service resource usage data.
- Identify relevant features for anomaly classification.
- Build a binary classification model.
- Compare Logistic Regression and Decision Tree models.
- Evaluate the models using classification metrics.
- Develop a simple Streamlit dashboard for real-time prediction.

## 📊 Dataset

The project uses a synthetic cloud microservices time-series dataset.

The final selected dataset contains:

- **10,080 observations**
- **8 input features**
- **1 target variable**
- **No missing values**
- **No duplicate rows**

### Selected Features

1. Node 1 CPU Utilization
2. Node 2 CPU Utilization
3. Carts DB Memory Utilization
4. User Network RX Bytes
5. Carts Request Rate
6. User Request Rate
7. Orders p95 Response Time
8. Front-end p95 Response Time

### Target

`anomaly`

- `0` → Normal
- `1` → Anomalous

## 🤖 Machine Learning Models

Two classification models were evaluated:

### Logistic Regression

- Class weighting: Balanced
- StandardScaler preprocessing
- Chronological 80/20 train-test split

### Decision Tree

- Maximum depth: 5
- Class weighting: Balanced

## 📈 Model Results

| Metric | Logistic Regression | Decision Tree |
|---|---:|---:|
| Accuracy | 96.33% | 95.44% |
| Precision | 86.30% | 71.60% |
| Recall | 49.61% | 45.67% |
| F1 Score | 63.00% | 55.77% |
| ROC-AUC | 88.20% | 79.60% |

The Logistic Regression model was used in the Streamlit application.

## 🖥️ Streamlit Application

The application allows users to enter cloud resource metrics and receive:

- Normal / Anomalous prediction
- Anomaly probability
- Resource profile information

### Example

A test case with elevated CPU, network activity, request rates, and response latency produced:

**Prediction:** Anomalous  
**Anomaly Probability:** 99.38%

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Google Colab
- GitHub

## 📁 Project Structure

```text
cloud-service-resource-anomaly-classification/
│
├── app.py
├── cloud_anomaly_model.pkl
├── cloud_anomaly_selected.csv
├── requirements.txt
├── README.md
└── .gitignore
