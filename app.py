import streamlit as st
import pandas as pd
import joblib


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model_package = joblib.load("cloud_anomaly_model.pkl")

model = model_package["model"]
scaler = model_package["scaler"]
feature_columns = model_package["features"]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cloud Anomaly Detection",
    page_icon="☁️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ---------- MAIN PAGE ---------- */

    .stApp {
        background-color: #0b1120;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ---------- HEADER ---------- */

    .hero {
        background: linear-gradient(
            135deg,
            #111c35 0%,
            #172f52 100%
        );

        border: 1px solid #29466b;
        border-radius: 20px;

        padding: 32px 38px;
        margin-bottom: 28px;
    }

    .hero-title {
        color: #f8fafc;
        font-size: 36px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 16px;
        line-height: 1.6;
    }


    /* ---------- SECTION HEADINGS ---------- */

    .section-title {
        color: #e2e8f0;
        font-size: 23px;
        font-weight: 700;
        margin-top: 24px;
        margin-bottom: 15px;
    }


    /* ---------- METRIC CARDS ---------- */

    div[data-testid="metric-container"] {
        background-color: #111827;
        border: 1px solid #263244;
        border-radius: 15px;
        padding: 18px;
    }


    /* ---------- RESULT CARDS ---------- */

    .normal-card {
        background: linear-gradient(
            135deg,
            #063b2a,
            #0b5139
        );

        border: 1px solid #16845c;
        border-radius: 20px;

        padding: 32px;
        text-align: center;

        margin-top: 15px;
    }


    .anomaly-card {
        background: linear-gradient(
            135deg,
            #481616,
            #681d1d
        );

        border: 1px solid #d64545;
        border-radius: 20px;

        padding: 32px;
        text-align: center;

        margin-top: 15px;
    }


    .result-title {
        color: white;
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 10px;
    }


    .probability {
        color: white;
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 8px;
    }


    .result-description {
        color: #dbeafe;
        font-size: 15px;
    }


    /* ---------- INFORMATION CARDS ---------- */

    .info-card {
        background-color: #111827;
        border: 1px solid #263244;
        border-radius: 15px;

        padding: 20px;
        min-height: 130px;
    }

    .info-title {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .info-value {
        color: #f8fafc;
        font-size: 22px;
        font-weight: 700;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;

        margin-top: 45px;
        padding-top: 20px;

        border-top: 1px solid #1e293b;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO HEADER
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-title">
        ☁️ Cloud Service Resource Anomaly Detection
    </div>

    <div class="hero-subtitle">
        AI-powered monitoring system for identifying unusual
        resource and service-performance behavior in cloud environments.
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">📊 Model Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🤖 Model",
        "Logistic Regression"
    )

with col2:
    st.metric(
        "📌 Input Features",
        "8"
    )

with col3:
    st.metric(
        "🎯 ROC-AUC",
        "88.20%"
    )

with col4:
    st.metric(
        "⚠️ Dataset Anomaly Rate",
        "7.65%"
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">🖥️ Cloud Resource Monitoring</div>',
    unsafe_allow_html=True
)

left_column, right_column = st.columns(2)


# ============================================================
# COMPUTE RESOURCES
# ============================================================

with left_column:

    st.markdown("### 🖥️ Compute Resources")

    cpu_node1 = st.number_input(
        "Node 1 CPU Utilization",
        min_value=0.0,
        value=10.0,
        step=0.1
    )

    cpu_node2 = st.number_input(
        "Node 2 CPU Utilization",
        min_value=0.0,
        value=5.0,
        step=0.1
    )

    memory_carts_db = st.number_input(
        "Carts DB Memory Utilization",
        min_value=0.0,
        value=5.0,
        step=0.1
    )


# ============================================================
# NETWORK AND TRAFFIC
# ============================================================

with right_column:

    st.markdown("### 🌐 Network & Traffic")

    network_user = st.number_input(
        "User Network RX Bytes",
        min_value=0.0,
        value=20000.0,
        step=100.0
    )

    carts_req_rate = st.number_input(
        "Carts Request Rate",
        min_value=0.0,
        value=5.0,
        step=0.1
    )

    user_req_rate = st.number_input(
        "User Request Rate",
        min_value=0.0,
        value=11.0,
        step=0.1
    )


# ============================================================
# SERVICE PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">⏱️ Service Performance</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    orders_p95 = st.number_input(
        "Orders p95 Response Time",
        min_value=0.0,
        value=80.0,
        step=1.0
    )

with col2:

    frontend_p95 = st.number_input(
        "Front-end p95 Response Time",
        min_value=0.0,
        value=170.0,
        step=1.0
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

predict = st.button(
    "🔍 Analyze Resource Behavior",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    # Create input DataFrame
    input_data = pd.DataFrame(
        [[
            cpu_node1,
            cpu_node2,
            memory_carts_db,
            network_user,
            carts_req_rate,
            user_req_rate,
            orders_p95,
            frontend_p95
        ]],
        columns=feature_columns
    )

    # Apply the same scaler used during model training
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Calculate anomaly probability
    probability = model.predict_proba(input_scaled)[0][1]


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">🚨 Prediction Result</div>',
        unsafe_allow_html=True
    )


    if prediction == 1:

        st.markdown(
            f"""
            <div class="anomaly-card">

                <div class="result-title">
                    ⚠️ ANOMALY DETECTED
                </div>

                <div class="probability">
                    {probability:.2%}
                </div>

                <div class="result-description">
                    High probability of anomalous cloud-resource behavior
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="normal-card">

                <div class="result-title">
                    ✅ SYSTEM NORMAL
                </div>

                <div class="probability">
                    {probability:.2%}
                </div>

                <div class="result-description">
                    Low probability of anomalous cloud-resource behavior
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PROBABILITY
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📈 Anomaly Probability")

    st.progress(
        int(probability * 100)
    )

    st.caption(
        f"Model-estimated anomaly probability: {probability:.2%}"
    )


    # ========================================================
    # RESOURCE PROFILE
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📊 Current Resource Profile")

    profile = pd.DataFrame({
        "Metric": [
            "Node 1 CPU",
            "Node 2 CPU",
            "Carts DB Memory",
            "User Network RX",
            "Carts Request Rate",
            "User Request Rate",
            "Orders p95 Response Time",
            "Front-end p95 Response Time"
        ],

        "Value": [
            cpu_node1,
            cpu_node2,
            memory_carts_db,
            network_user,
            carts_req_rate,
            user_req_rate,
            orders_p95,
            frontend_p95
        ]
    })

    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ☁️ Cloud AI Monitor")

    st.markdown("---")

    st.markdown("### 📌 About")

    st.write(
        "This educational machine-learning prototype "
        "classifies cloud-service observations as "
        "Normal or Anomalous."
    )

    st.markdown("### 🧠 Model")

    st.write("Logistic Regression")

    st.markdown("### 📊 Test Performance")

    st.write("Accuracy: 96.33%")
    st.write("Precision: 86.30%")
    st.write("Recall: 49.61%")
    st.write("F1 Score: 63.00%")
    st.write("ROC-AUC: 88.20%")

    st.markdown("---")

    st.caption(
        "Educational ML prototype"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    Cloud Service Resource Anomaly Classification
    • Machine Learning Capstone Project

</div>
""", unsafe_allow_html=True)
