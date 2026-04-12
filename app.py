import streamlit as st
import pandas as pd
import pickle
import plotly.express as px


st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    layout="wide"
)


st.markdown("""
<style>
    body {
        background-color: #f8f9fc;
    }

    .main-title {
        text-align: center;
        font-size: 170px;
        font-weight: bold;
        color: #2c3e50;
    }

    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Customer Intelligence Dashboard</p>', unsafe_allow_html=True)
st.caption("AI-powered decision system for customer segmentation and business strategy")

model = pickle.load(open('customer_segmentation_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
customer_df = pd.read_csv("customer_data.csv")


st.sidebar.markdown("🔍 Customer Input Panel")

total_spent = st.sidebar.slider(
    " Total Spent",
    0,
    int(customer_df['TotalSpent'].max()),
    value=5000,
    step=1000
)

total_orders = st.sidebar.slider(
    " Total Orders",
    0,
    int(customer_df['TotalOrders'].max()),
    value=10,
    step=5
)

cluster = None

if st.sidebar.button("Predict Cluster"):

    new_customer = pd.DataFrame(
        [[total_spent, total_orders]],
        columns=["TotalSpent", "TotalOrders"]
    )

    new_customer_scaled = scaler.transform(new_customer)
    prediction = model.predict(new_customer_scaled)

    cluster = prediction[0]

    st.sidebar.success(f"Customer belongs to Cluster {cluster}")


    if cluster == 0:
        st.sidebar.info("Offer discounts and campaigns to increase engagement.")

    elif cluster == 1:
        st.sidebar.success("Target with premium products and exclusive deals.")

    elif cluster == 2:
        st.sidebar.info("Introduce loyalty programs and reward points.")

    elif cluster == 3:
        st.sidebar.warning("Use retargeting ads to bring them back.")

    elif cluster == 4:
        st.sidebar.success("Provide VIP services and early access offers.")


st.markdown("Business Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(customer_df))
col2.metric("Avg Spending", round(customer_df['TotalSpent'].mean(), 2))
col3.metric("Avg Orders", round(customer_df['TotalOrders'].mean(), 2))


tab1, tab2, tab3 = st.tabs(["Overview", "Visualization", "Insights"])


with tab1:
    st.subheader("Key Business Insights")

    st.write(f"Total Customers: {len(customer_df)}")
    st.write(f"Average Spending: {round(customer_df['TotalSpent'].mean(), 2)}")
    st.write(f"Average Orders: {round(customer_df['TotalOrders'].mean(), 2)}")

    st.write("Customers are segmented based on spending behavior and purchase frequency.")


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

    st.plotly_chart(fig, width='stretch')

    st.subheader("Cluster Distribution")

    cluster_counts = customer_df['Cluster'].value_counts()
    st.bar_chart(cluster_counts)

with tab3:
    st.subheader("Business Insights")

    st.markdown("""
    🔹 **Cluster 0** → Low-value customers  
    🔹 **Cluster 1** → High spenders  
    🔹 **Cluster 2** → Frequent buyers  
    🔹 **Cluster 3** → Occasional buyers  
    🔹 **Cluster 4** → Premium customers  

    **Business Strategies:**
    - Target high spenders with premium offers  
    - Retain frequent buyers with loyalty programs  
    - Convert low-value customers using discounts  
    """)


if cluster is not None:
    st.subheader("Business Recommendation")

    if cluster == 0:
        st.write("Low-value customers → Use discounts and campaigns.")

    elif cluster == 1:
        st.write("High-value customers → Focus on retention and premium services.")

    elif cluster == 2:
        st.write("Frequent buyers → Use loyalty programs.")

    elif cluster == 3:
        st.write("Occasional customers → Apply retargeting ads.")

    elif cluster == 4:
        st.write("Premium customers → Offer VIP experience.")


st.markdown("---")
st.subheader("AI Business Assistant")

user_input = st.text_input("Ask business questions about customers")

if user_input:
    user_input = user_input.lower()

    if "high value" in user_input:
        st.success("Focus on Cluster 1 & 4 customers for maximum revenue.")

    elif "low value" in user_input:
        st.info("Use discounts to convert low-value customers.")

    elif "frequent" in user_input:
        st.success("Loyalty programs work best for frequent buyers.")

    elif "increase sales" in user_input:
        st.write("Target premium + frequent customers to boost revenue.")

    elif "recommend" in user_input:
        st.write("Focus on high spenders and frequent buyers.")

    else:
        st.warning("Try asking about strategies or customer types.")

