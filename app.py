import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Cloud Anomaly Monitor",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    package = joblib.load("cloud_anomaly_model.pkl")

    model = package["model"]
    scaler = package["scaler"]
    features = package["features"]

    return model, scaler, features


model, scaler, feature_columns = load_model()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("☁️ Cloud AI Monitor")

    st.markdown("---")

    st.subheader("🧠 Model")

    st.write("**Logistic Regression**")

    st.markdown("---")

    st.subheader("📊 Test Performance")

    st.metric("Accuracy", "96.33%")
    st.metric("Precision", "86.30%")
    st.metric("Recall", "49.61%")
    st.metric("F1 Score", "63.00%")
    st.metric("ROC-AUC", "88.20%")

    st.markdown("---")

    st.subheader("📁 Dataset")

    st.write("10,080 observations")
    st.write("8 selected features")
    st.write("Binary anomaly classification")

    st.markdown("---")

    st.caption(
        "Educational machine-learning prototype "
        "for cloud-service resource anomaly detection."
    )


# =========================================================
# HEADER
# =========================================================

st.title("☁️ Cloud Service Resource Anomaly Detection")

st.write(
    "Monitor cloud resource behavior and classify unusual "
    "service conditions using machine learning."
)

st.markdown("---")


# =========================================================
# MODEL OVERVIEW
# =========================================================

st.header("📊 Model Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Model",
        "Logistic Regression"
    )

with col2:
    st.metric(
        "Input Features",
        "8"
    )

with col3:
    st.metric(
        "ROC-AUC",
        "88.20%"
    )

with col4:
    st.metric(
        "Dataset Anomaly Rate",
        "7.65%"
    )


st.markdown("---")


# =========================================================
# RESOURCE INPUTS
# =========================================================

st.header("🎛️ Resource Monitoring")

st.write(
    "Enter the current cloud-service measurements "
    "and analyze the resource behavior."
)


# =========================================================
# COMPUTE RESOURCES
# =========================================================

st.subheader("🖥️ Compute Resources")

col1, col2, col3 = st.columns(3)

with col1:

    cpu_node1 = st.number_input(
        "Node 1 CPU Utilization",
        min_value=0.0,
        value=10.0,
        step=0.5
    )

with col2:

    cpu_node2 = st.number_input(
        "Node 2 CPU Utilization",
        min_value=0.0,
        value=5.0,
        step=0.5
    )

with col3:

    memory = st.number_input(
        "Carts DB Memory Utilization",
        min_value=0.0,
        value=4.5,
        step=0.5
    )


# =========================================================
# NETWORK AND TRAFFIC
# =========================================================

st.subheader("🌐 Network & Traffic")

col1, col2, col3 = st.columns(3)

with col1:

    network = st.number_input(
        "User Network RX Bytes",
        min_value=0.0,
        value=20000.0,
        step=500.0
    )

with col2:

    carts_rate = st.number_input(
        "Carts Request Rate",
        min_value=0.0,
        value=5.5,
        step=0.5
    )

with col3:

    user_rate = st.number_input(
        "User Request Rate",
        min_value=0.0,
        value=11.0,
        step=0.5
    )


# =========================================================
# SERVICE PERFORMANCE
# =========================================================

st.subheader("⚡ Service Performance")

col1, col2 = st.columns(2)

with col1:

    orders_latency = st.number_input(
        "Orders p95 Response Time",
        min_value=0.0,
        value=80.0,
        step=10.0
    )

with col2:

    frontend_latency = st.number_input(
        "Front-end p95 Response Time",
        min_value=0.0,
        value=170.0,
        step=10.0
    )


st.markdown("")


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze = st.button(
    "🔍 Analyze Resource Behavior",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if analyze:

    # -----------------------------------------------------
    # CREATE INPUT DATAFRAME
    # -----------------------------------------------------

    input_data = pd.DataFrame(
        [[
            cpu_node1,
            cpu_node2,
            memory,
            network,
            carts_rate,
            user_rate,
            orders_latency,
            frontend_latency
        ]],
        columns=feature_columns
    )


    # -----------------------------------------------------
    # SCALE INPUT
    # -----------------------------------------------------

    scaled_data = scaler.transform(input_data)


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(scaled_data)[0]

    probability = model.predict_proba(scaled_data)[0][1]

    probability_percent = probability * 100


    # =====================================================
    # DETECTION RESULT
    # =====================================================

    st.markdown("---")

    st.header("🎯 Detection Result")


    if prediction == 1:

        st.error(
            "🚨 ANOMALOUS\n\n"
            "Unusual resource or service behavior detected."
        )

    else:

        st.success(
            "✅ NORMAL\n\n"
            "Resource behavior appears within the learned "
            "normal pattern."
        )


    # =====================================================
    # ANOMALY PROBABILITY
    # =====================================================

    st.subheader("🎯 Anomaly Probability")


    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_percent,
            number={
                "suffix": "%",
                "font": {
                    "size": 34
                }
            },
            title={
                "text": "Probability of Anomalous Behavior"
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
                        "color": "#dbeafe"
                    },
                    {
                        "range": [50, 75],
                        "color": "#fef3c7"
                    },
                    {
                        "range": [75, 100],
                        "color": "#fee2e2"
                    }
                ]
            }
        )
    )


    gauge.update_layout(
        height=330,
        margin=dict(
            l=30,
            r=30,
            t=60,
            b=20
        ),
        paper_bgcolor="rgba(0,0,0,0)"
    )


    st.plotly_chart(
        gauge,
        use_container_width=True
    )


    # =====================================================
    # CURRENT RESOURCE PROFILE
    # =====================================================

    st.markdown("---")

    st.header("📈 Current Resource Profile")


    profile = pd.DataFrame(
        {
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
        }
    )


    fig_profile = px.bar(
        profile,
        x="Metric",
        y="Value",
        title="Current Cloud Resource Measurements"
    )


    fig_profile.update_layout(
        height=450,
        xaxis_tickangle=-35,
        margin=dict(
            l=30,
            r=30,
            t=70,
            b=110
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )


    st.plotly_chart(
        fig_profile,
        use_container_width=True
    )


    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    st.markdown("---")

    st.header("📊 Model Performance")


    performance = pd.DataFrame(
        {
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
        }
    )


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
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )


    st.plotly_chart(
        fig_performance,
        use_container_width=True
    )


    # =====================================================
    # FEATURE IMPACT
    # =====================================================

    st.markdown("---")

    st.header("🧠 Feature Impact")

    st.write(
        "Logistic regression coefficients show the direction "
        "and relative strength of each selected feature."
    )


    coefficients = model.coef_[0]


    feature_impact = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Impact": coefficients
        }
    ).sort_values(
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
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )


    st.plotly_chart(
        fig_impact,
        use_container_width=True
    )


    # =====================================================
    # INPUT SUMMARY
    # =====================================================

    st.markdown("---")

    st.header("📋 Input Summary")


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


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Cloud Service Resource Anomaly Classification • "
    "Educational ML Prototype • Logistic Regression"
)
