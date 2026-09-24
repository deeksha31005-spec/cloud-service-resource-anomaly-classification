import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Cloud Anomaly Monitor",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

    /* Main application */
    .stApp {
        background: #0b0f19;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }

    /* Main content */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* Headings */
    h1, h2, h3 {
        letter-spacing: -0.5px;
    }

    /* Hero */
    .hero {
        padding: 28px 32px;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #111827 0%,
            #172554 100%
        );
        border: 1px solid #263449;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #9ca3af;
        font-size: 16px;
    }

    /* Cards */
    .card {
        padding: 20px;
        border-radius: 16px;
        background: #111827;
        border: 1px solid #263449;
        margin-bottom: 18px;
    }

    .card-title {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    /* Prediction cards */
    .prediction-normal {
        padding: 28px;
        border-radius: 18px;
        background: #062e1b;
        border: 1px solid #166534;
        text-align: center;
    }

    .prediction-anomaly {
        padding: 28px;
        border-radius: 18px;
        background: #3b0a0a;
        border: 1px solid #991b1b;
        text-align: center;
    }

    .prediction-title {
        font-size: 30px;
        font-weight: 700;
    }

    .prediction-subtitle {
        color: #d1d5db;
        margin-top: 6px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #263449;
        padding: 16px;
        border-radius: 14px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3.2em;
        font-weight: 600;
        border: none;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #6d28d9);
        color: white;
    }

    /* Divider */
    hr {
        border-color: #263449;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        padding-top: 25px;
        font-size: 13px;
    }

    /* Light mode support */
    @media (prefers-color-scheme: light) {

        .stApp {
            background: #f7f9fc;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e5e7eb;
        }

        .hero {
            background: linear-gradient(
                135deg,
                #ffffff 0%,
                #eef4ff 100%
            );
            border: 1px solid #dbe3ef;
        }

        .hero-subtitle {
            color: #64748b;
        }

        .card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
        }

        hr {
            border-color: #e2e8f0;
        }

        .footer {
            color: #64748b;
        }
    }

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    package = joblib.load("cloud_anomaly_model.pkl")

    return (
        package["model"],
        package["scaler"],
        package["features"]
    )


model, scaler, feature_columns = load_model()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.markdown("## ☁️ Cloud AI Monitor")

    st.markdown("---")

    st.markdown("### 🧠 Model")

    st.write("Logistic Regression")

    st.markdown("### 📊 Test Performance")

    st.metric("Accuracy", "96.33%")
    st.metric("Precision", "86.30%")
    st.metric("Recall", "49.61%")
    st.metric("F1 Score", "63.00%")
    st.metric("ROC-AUC", "88.20%")

    st.markdown("---")

    st.caption(
        "Educational machine-learning prototype "
        "for cloud resource anomaly classification."
    )

# ---------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------

st.markdown("""
<div class="hero">

    <div class="hero-title">
        ☁️ Cloud Service Resource Anomaly Detection
    </div>

    <div class="hero-subtitle">
        Monitor cloud resource behavior and classify unusual
        service conditions using machine learning.
    </div>

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL OVERVIEW
# ---------------------------------------------------------

st.subheader("📊 Model Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Model",
        "Logistic Regression"
    )

with c2:
    st.metric(
        "Input Features",
        "8"
    )

with c3:
    st.metric(
        "ROC-AUC",
        "88.20%"
    )

with c4:
    st.metric(
        "Anomaly Rate",
        "7.65%"
    )

# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------

st.markdown("---")

st.subheader("🎛️ Resource Monitoring")

st.caption(
    "Enter the current cloud-service resource measurements "
    "to evaluate whether the observation is anomalous."
)

# Compute
st.markdown("### 🖥️ Compute Resources")

c1, c2, c3 = st.columns(3)

with c1:
    cpu_node1 = st.number_input(
        "Node 1 CPU Utilization",
        min_value=0.0,
        value=10.0,
        step=0.5
    )

with c2:
    cpu_node2 = st.number_input(
        "Node 2 CPU Utilization",
        min_value=0.0,
        value=5.0,
        step=0.5
    )

with c3:
    memory = st.number_input(
        "Carts DB Memory Utilization",
        min_value=0.0,
        value=4.5,
        step=0.5
    )

# Network
st.markdown("### 🌐 Network & Traffic")

c1, c2, c3 = st.columns(3)

with c1:
    network = st.number_input(
        "User Network RX Bytes",
        min_value=0.0,
        value=20000.0,
        step=500.0
    )

with c2:
    carts_rate = st.number_input(
        "Carts Request Rate",
        min_value=0.0,
        value=5.5,
        step=0.5
    )

with c3:
    user_rate = st.number_input(
        "User Request Rate",
        min_value=0.0,
        value=11.0,
        step=0.5
    )

# Service performance
st.markdown("### ⚡ Service Performance")

c1, c2 = st.columns(2)

with c1:
    orders_latency = st.number_input(
        "Orders p95 Response Time",
        min_value=0.0,
        value=80.0,
        step=10.0
    )

with c2:
    frontend_latency = st.number_input(
        "Front-end p95 Response Time",
        min_value=0.0,
        value=170.0,
        step=10.0
    )

# ---------------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------------

st.markdown("")

analyze = st.button(
    "🔍 Analyze Resource Behavior"
)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if analyze:

    input_data = pd.DataFrame([[
        cpu_node1,
        cpu_node2,
        memory,
        network,
        carts_rate,
        user_rate,
        orders_latency,
        frontend_latency
    ]], columns=feature_columns)

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]

    probability = model.predict_proba(scaled_data)[0][1]

    probability_percent = probability * 100

    st.markdown("---")

    st.subheader("🎯 Detection Result")

    if prediction == 1:

        st.markdown(
            f"""
            <div class="prediction-anomaly">

                <div class="prediction-title">
                    🚨 ANOMALOUS
                </div>

                <div class="prediction-subtitle">
                    Unusual resource or service behavior detected
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="prediction-normal">

                <div class="prediction-title">
                    ✅ NORMAL
                </div>

                <div class="prediction-subtitle">
                    Resource behavior appears within the learned normal pattern
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # PROBABILITY GAUGE
    # -----------------------------------------------------

    st.markdown("### 🎯 Anomaly Probability")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_percent,
            number={
                "suffix": "%",
                "font": {"size": 34}
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "bar": {
                    "color": "#7c3aed"
                },
                "steps": [
                    {
                        "range": [0, 50],
                        "color": "#172033"
                    },
                    {
                        "range": [50, 75],
                        "color": "#332a12"
                    },
                    {
                        "range": [75, 100],
                        "color": "#351414"
                    }
                ]
            }
        )
    )

    gauge.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    # -----------------------------------------------------
    # RESOURCE PROFILE
    # -----------------------------------------------------

    st.markdown("### 📈 Current Resource Profile")

    profile = pd.DataFrame({
        "Metric": [
            "Node 1 CPU",
            "Node 2 CPU",
            "Carts DB Memory",
            "User Network RX",
            "Carts Request Rate",
            "User Request Rate",
            "Orders p95",
            "Front-end p95"
        ],
        "Value": [
            cpu_node1,
            cpu_node2,
            memory,
            network,
            carts_rate,
            user_rate,
            orders_latency,
            frontend_latency
        ]
    })

    fig_profile = px.bar(
        profile,
        x="Metric",
        y="Value",
        title="Current Cloud Resource Measurements"
    )

    fig_profile.update_layout(
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-35,
        margin=dict(l=20, r=20, t=60, b=100)
    )

    st.plotly_chart(
        fig_profile,
        use_container_width=True
    )

    # -----------------------------------------------------
    # MODEL PERFORMANCE
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader("📊 Model Performance")

    performance = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],
        "Score": [
            96.33,
            86.30,
            49.61,
            63.00,
            88.20
        ]
    })

    fig_performance = px.bar(
        performance,
        x="Metric",
        y="Score",
        text="Score",
        title="Logistic Regression Test Performance"
    )

    fig_performance.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_performance.update_yaxes(
        range=[0, 105],
        title="Score (%)"
    )

    fig_performance.update_layout(
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig_performance,
        use_container_width=True
    )

    # -----------------------------------------------------
    # FEATURE IMPACT
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader("🧠 Feature Impact")

    coefficients = model.coef_[0]

    feature_impact = pd.DataFrame({
        "Feature": feature_columns,
        "Impact": coefficients
    }).sort_values(
        "Impact",
        ascending=True
    )

    fig_impact = px.bar(
        feature_impact,
        x="Impact",
        y="Feature",
        orientation="h",
        title="Logistic Regression Feature Coefficients"
    )

    fig_impact.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig_impact,
        use_container_width=True
    )

    # -----------------------------------------------------
    # INPUT TABLE
    # -----------------------------------------------------

    st.subheader("📋 Input Summary")

    display_data = input_data.T.reset_index()

    display_data.columns = [
        "Feature",
        "Input Value"
    ]

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        Cloud Service Resource Anomaly Classification •
        Educational ML Prototype • Logistic Regression
    </div>
    """,
    unsafe_allow_html=True
)
