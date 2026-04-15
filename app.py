import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

st.set_page_config(
    page_title="AI Customer Profiler",
    layout="centered"
)

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background-color: #eef2f7 !important;
}

body {
    background-color: #eef2f7 !important;
}

.main-title {
    text-align: center;
    font-size: 55px;
    font-weight: 900;
    color: #0f172a;
}

.tagline {
    text-align: center;
    font-size: 18px;
    color: #64748b;
    margin-bottom: 40px;
}

.prediction-card {
    padding: 40px;
    border-radius: 25px;
    color: white;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin: 20px 0;
}

.card-title {
    font-size: 32px;
    font-weight: 800;
}

.card-subtitle {
    font-size: 18px;
    opacity: 0.9;
}

.bg-low { background: linear-gradient(135deg, #FF9D6C, #BB4E75); }
.bg-high { background: linear-gradient(135deg, #11998e, #38ef7d); }
.bg-frequent { background: linear-gradient(135deg, #00c6ff, #0072ff); }
.bg-occasional { background: linear-gradient(135deg, #8E2DE2, #4A00E0); }
.bg-premium { background: linear-gradient(135deg, #FFD700, #B8860B); }

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = pickle.load(open('customer_segmentation_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    df = pd.read_csv("customer_data.csv")
    return model, scaler, df

model, scaler, customer_df = load_assets()

st.markdown('<p class="main-title">Customer AI</p>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Real-time Segmentation & Behavioral Strategy Engine</p>', unsafe_allow_html=True)

tab_main, tab_insights = st.tabs(["Prediction Console", "Growth Analytics"])

with tab_main:

    st.markdown("### Input Customer Metrics")

    c1, c2 = st.columns(2)
    with c1:
        spent = st.number_input("Total Revenue Contribution ($)", min_value=0, value=2500, step=500)
    with c2:
        orders = st.number_input("Total Purchase Frequency", min_value=0, value=5, step=1)

    if st.button("EXECUTE ANALYSIS", use_container_width=True):

        input_data = pd.DataFrame([[spent, orders]], columns=["TotalSpent", "TotalOrders"])
        scaled_data = scaler.transform(input_data)
        cluster = model.predict(scaled_data)[0]

        segments = {
            0: {
                "name": "Low-Value",
                "class": "bg-low",
                "explanation": "This customer shows low spending and low engagement. They are at risk of churn and may not see enough value in your product yet.",
                "tips": [
                    "Launch re-engagement campaigns using email or SMS with personalized offers based on their past activity.",
                    "Provide limited-time discounts or entry-level bundles to encourage their next purchase.",
                    "Analyze where they dropped off and remove friction in onboarding or checkout."
                ]
            },
            1: {
                "name": "High-Value",
                "class": "bg-high",
                "explanation": "This customer contributes significant revenue and is highly valuable to the business. Retaining them is critical.",
                "tips": [
                    "Offer premium support or dedicated assistance to enhance their experience.",
                    "Provide exclusive deals, early access, and loyalty rewards to maintain satisfaction.",
                    "Monitor behavior patterns and proactively engage to prevent churn."
                ]
            },
            2: {
                "name": "Frequent Buyer",
                "class": "bg-frequent",
                "explanation": "This customer purchases regularly and shows strong engagement but may have moderate spending per order.",
                "tips": [
                    "Introduce loyalty programs with reward points or cashback incentives.",
                    "Use personalized recommendations to increase average order value.",
                    "Bundle products or upsell to maximize revenue from frequent purchases."
                ]
            },
            3: {
                "name": "Occasional",
                "class": "bg-occasional",
                "explanation": "This customer interacts irregularly and lacks consistent purchasing behavior.",
                "tips": [
                    "Run retargeting ads and timely email reminders to re-engage them.",
                    "Send personalized notifications based on browsing or past purchases.",
                    "Create urgency using limited-time offers or seasonal campaigns."
                ]
            },
            4: {
                "name": "Premium",
                "class": "bg-premium",
                "explanation": "This is a top-tier customer with very high spending and strong loyalty. They are key revenue drivers.",
                "tips": [
                    "Provide VIP experiences such as exclusive memberships or concierge services.",
                    "Offer early access to products and personalized premium deals.",
                    "Encourage referrals and testimonials to leverage their influence."
                ]
            }
        }

        res = segments.get(cluster)

        st.markdown(f"""
        <div class="prediction-card {res['class']}">
            <div class="card-title">{res['name']} Customer</div>
            <div class="card-subtitle">AI-powered segmentation result</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### What this means")

        st.markdown(f"""
        <div style="padding:20px; border-radius:15px; background:#f8fafc;">
        <b>Customer Segment:</b> {res['name']} <br><br>
        {res['explanation']} <br><br>
        <b>Business Impact:</b> This classification helps you decide how to engage this customer effectively, increase retention, boost revenue, or prevent churn.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Recommendations")

        col1, col2, col3 = st.columns(3)
        for i, tip in enumerate(res['tips']):
            with [col1, col2, col3][i]:
                st.info(f"Step {i+1}\n\n{tip}")

with tab_insights:

    st.subheader("Customer Segmentation Map")

    fig = px.scatter(
        customer_df,
        x="TotalSpent",
        y="TotalOrders",
        color="Cluster",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Customer Segment Breakdown")

    st.markdown("""
    <div style="padding:25px; border-radius:20px; background:white;">
    <b>Low-Value Customers</b> → Low spending, low engagement <br>
    <b>High-Value Customers</b> → High spending <br>
    <b>Frequent Buyers</b> → Regular purchases <br>
    <b>Occasional Customers</b> → Irregular buying <br>
    <b>Premium Customers</b> → Very high-value customers  
    </div>
    """, unsafe_allow_html=True)