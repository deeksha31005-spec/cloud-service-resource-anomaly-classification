# Cloud Service Resource Anomaly Classification

A machine learning-based system for identifying unusual resource-usage observations in a simulated cloud-service environment.

The project analyzes cloud resource and service-performance metrics and classifies observations as:

- ✅ Normal
- ⚠️ Anomalous

The complete workflow includes data analysis, preprocessing, machine learning model development, model comparison, evaluation, trained-model export, and an interactive Streamlit application.

---

## 📌 Project Overview

Cloud services generate large amounts of resource and performance data. Detecting unusual behavior from these metrics can help identify potentially anomalous operating periods.

This project investigates whether selected cloud resource and service-performance metrics can distinguish between normal and anomalous observations.

The project follows an end-to-end machine learning workflow:

**Dataset → EDA → Feature Selection → Preprocessing → Model Training → Evaluation → Model Comparison → Deployment**

---

## 🎯 Objectives

- Analyze cloud-service resource and performance data.
- Perform data-quality checks and exploratory data analysis.
- Identify relevant features for anomaly classification.
- Prepare the data for machine learning.
- Build a binary classification system.
- Evaluate Logistic Regression and Decision Tree models.
- Compare the performance of the evaluated models.
- Select a final model for deployment.
- Save the trained model and preprocessing information.
- Develop an interactive Streamlit dashboard for anomaly prediction.

---

## 📊 Dataset

The project uses a synthetic cloud microservices time-series dataset.

### Dataset Statistics

- **Observations:** 10,080
- **Input features:** 8
- **Target variable:** 1
- **Total columns:** 10 including time and target
- **Missing values:** 0
- **Duplicate rows:** 0

### Target Distribution

| Target | Meaning | Count |
|--------|---------|------:|
| `0` | Normal | 9,309 |
| `1` | Anomalous | 771 |

The observations are time-ordered. Therefore, a chronological train-test split was used.

---

## 🔍 Selected Features

The final dataset contains eight selected input features covering different cloud resource and service-performance categories:

1. Node 1 CPU Utilization
2. Node 2 CPU Utilization
3. Carts DB Memory Utilization
4. User Network RX Bytes
5. Carts Request Rate
6. User Request Rate
7. Orders p95 Response Time
8. Front-end p95 Response Time

### Target Variable

`anomaly`

- `0` → Normal
- `1` → Anomalous

---

## 🔄 Machine Learning Workflow

The project follows these steps:

1. Load the selected dataset.
2. Check missing values and duplicate rows.
3. Analyze the target class distribution.
4. Perform exploratory data analysis.
5. Analyze relationships between selected features.
6. Split the data chronologically.
7. Standardize the features for Logistic Regression.
8. Train the classification models.
9. Evaluate both models.
10. Compare model performance.
11. Select the final model.
12. Save the trained model and scaler.
13. Deploy the final model using Streamlit.

---

## 🧹 Data Preprocessing

### Chronological Train-Test Split

An **80/20 chronological split** was used to preserve the time order of the observations.

- **Training observations:** 8,064
- **Testing observations:** 2,016

The test data represents later observations than the training data.

### Feature Scaling

`StandardScaler` was used for Logistic Regression.

The scaler was fitted using **training data only** and then applied to the test data.

---

## 🤖 Machine Learning Models

Two machine learning classification models were evaluated.

### 1. Logistic Regression

Configuration:

- Class weighting: `balanced`
- `max_iter`: 1000
- Random state: 42
- StandardScaler preprocessing

### 2. Decision Tree

Configuration:

- Maximum depth: 5
- Class weighting: `balanced`
- Random state: 42

---

## 📈 Model Comparison

Both models were evaluated using the same chronological test set.

| Metric | Logistic Regression | Decision Tree |
|--------|--------------------:|--------------:|
| Accuracy | **96.33%** | 95.44% |
| Precision | **86.30%** | 71.60% |
| Recall | **49.61%** | 45.67% |
| F1 Score | **63.00%** | 55.77% |
| ROC-AUC | **88.20%** | 79.60% |

### Final Model

**Logistic Regression** was selected as the final model for the Streamlit application based on the evaluation results.

The Decision Tree was used as a comparison model during the machine learning experimentation, but it is **not used for the final Streamlit prediction**.

---

## 📊 Logistic Regression Evaluation

The final Logistic Regression model achieved:

- **Accuracy:** 96.33%
- **Precision:** 86.30%
- **Recall:** 49.61%
- **F1 Score:** 63.00%
- **ROC-AUC:** 88.20%

### Confusion Matrix

| | Predicted Normal | Predicted Anomalous |
|---|---:|---:|
| **Actual Normal** | 1,879 | 10 |
| **Actual Anomalous** | 64 | 63 |

### ROC-AUC

The Logistic Regression model achieved an ROC-AUC of **88.20%** on the chronological test set.

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit dashboard for demonstrating the **final Logistic Regression model**.

The application allows users to enter cloud resource and service-performance measurements and obtain an anomaly prediction.

### Input Categories

#### Compute Resources

- Node 1 CPU Utilization
- Node 2 CPU Utilization
- Carts DB Memory Utilization

#### Network & Traffic

- User Network RX Bytes
- Carts Request Rate
- User Request Rate

#### Service Performance

- Orders p95 Response Time
- Front-end p95 Response Time

### Application Output

The dashboard provides:

- Normal / Anomalous prediction
- Anomaly probability
- Current resource profile
- Model performance metrics
- Confusion Matrix
- ROC Curve
- Feature impact visualization
- Input summary

### Deployed Model

The Streamlit application uses:

**Logistic Regression**

The Decision Tree is not used in the deployed application because it was evaluated as a comparison model during development.

---

## 🌐 Live Demo

Try the deployed Streamlit application:

**Cloud Service Resource Anomaly Classification Dashboard**

https://cloud-service-resource-anomaly-classification-dkqzkvtakffkt4m3.streamlit.app/
---



## 🧪 Example Prediction

An example observation with elevated resource usage, network activity, request rates, and response latency:

| Feature | Example Value |
|---------|--------------:|
| Node 1 CPU Utilization | 20 |
| Node 2 CPU Utilization | 10 |
| Carts DB Memory Utilization | 6 |
| User Network RX Bytes | 30,000 |
| Carts Request Rate | 8 |
| User Request Rate | 15 |
| Orders p95 Response Time | 500 |
| Front-end p95 Response Time | 450 |

### Result

**Prediction:** Anomalous

**Anomaly Probability:** 99.38%

---

## 📓 Jupyter / Google Colab Notebook

The complete machine learning workflow is available in:

`Cloud_Service_Resource_Anomaly_Classification (1).ipynb`

The notebook contains:

- Dataset loading
- Data-quality analysis
- Target distribution analysis
- Exploratory data analysis
- Feature relationship analysis
- Feature selection
- Chronological train-test split
- Feature scaling
- Logistic Regression
- Logistic Regression evaluation
- Decision Tree
- Decision Tree evaluation
- Model comparison
- Confusion Matrix
- ROC Curve
- Feature impact analysis
- Final model export

---

## 💾 Trained Model

The final trained model is stored in:

`cloud_anomaly_model.pkl`

The saved model package contains:

- Trained Logistic Regression model
- Fitted StandardScaler
- Selected feature names

The Streamlit application loads this file to generate predictions.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Joblib
- Streamlit
- Google Colab
- GitHub

---

## 📁 Project Structure

```text
cloud-service-resource-anomaly-classification/
│
├── app.py
├── cloud_anomaly_model.pkl
├── cloud_anomaly_selected.csv
├── Cloud_Service_Resource_Anomaly_Classification (1).ipynb
├── requirements.txt
├── README.md
└── .gitignore
