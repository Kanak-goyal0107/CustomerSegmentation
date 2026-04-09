import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# =========================
# CUSTOM STYLING
# =========================
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Customer Segmentation Dashboard</p>', unsafe_allow_html=True)

# =========================
# LOAD FILES
# =========================
model = pickle.load(open('customer_segmentation_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
customer_df = pd.read_csv("customer_data.csv")

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.title("🔍 Customer Input")
st.sidebar.markdown("---")

total_spent = st.sidebar.number_input("💰 Total Spent", min_value=0.0)
total_orders = st.sidebar.number_input("📦 Total Orders", min_value=0)

# =========================
# PREDICTION
# =========================
if st.sidebar.button("Predict Cluster"):
    new_customer = [[total_spent, total_orders]]
    new_customer_scaled = scaler.transform(new_customer)
    prediction = model.predict(new_customer_scaled)

    st.sidebar.success(f"🎯 Customer belongs to Cluster {prediction[0]}")

# =========================
# METRICS SECTION
# =========================
st.markdown("## 📊 Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(customer_df))
col2.metric("Avg Spending", round(customer_df['TotalSpent'].mean(), 2))
col3.metric("Avg Orders", round(customer_df['TotalOrders'].mean(), 2))

# =========================
# TABS SECTION
# =========================
tab1, tab2, tab3 = st.tabs(["📄 Data", "📈 Visualization", "📌 Insights"])

# -------- TAB 1 --------
with tab1:
    st.subheader("Customer Dataset")
    st.dataframe(customer_df)

# -------- TAB 2 --------
with tab2:
    st.subheader("Customer Segmentation Visualization")

    fig = px.scatter(
        customer_df,
        x="TotalSpent",
        y="TotalOrders",
        color="Cluster",
        title="Customer Segmentation",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Distribution")

    cluster_counts = customer_df['Cluster'].value_counts()
    st.bar_chart(cluster_counts)

# -------- TAB 3 --------
with tab3:
    st.subheader("Business Insights")

    st.markdown("""
    🔹 **Cluster 0** → Low-value customers  
    🔹 **Cluster 1** → High spenders  
    🔹 **Cluster 2** → Frequent buyers  
    🔹 **Cluster 3** → Occasional buyers  
    🔹 **Cluster 4** → Premium customers  

    📌 Businesses can:
    - Target high spenders with premium offers  
    - Retain frequent buyers with loyalty programs  
    - Convert low-value customers using discounts  
    """)

# =========================
# FOOTER
# =========================
st.markdown("---")
