import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(
    page_title = "Vendor Invoice Intelligence System",
    page_icon = "📊",
    layout = "wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("../data/cleaned_supply_chain_data.csv")
df = load_data()

low_threshold = df["Estimated_Freight_Cost"].quantile(0.33)
medium_threshold = df["Estimated_Freight_Cost"].quantile(0.66)
print(low_threshold)
print(medium_threshold)

@st.cache_resource
def load_freight_model():
    return joblib.load("../models/freight_model.pkl")
freight_model = load_freight_model()

@st.cache_resource
def load_risk_model():
    return joblib.load("../models/risk_model.pkl")
risk_model = load_risk_model()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "",
    [
        "🏠 Home",
        "📊 Dataset Insights",
        "🚚 Freight Cost Prediction",
        "⚠️ Risk Prediction",
        "📈 Business Analytics"
    ]
)

# Home page
if page == "🏠 Home":
    # Title and description
    st.title("📦 Vendor Invoice Intelligence System")

    st.markdown("""
    ### AI-Powered Freight Cost Prediction and Invoice Risk Detection
                
    This dashboard helps organizations:
                
    - Predict Freight Cost
    - Detect Risky Invoices
    - Analyze Logistics Performance
    - Generate Buisness Insights
                
    Built using:
    - Python
    - Padas
    - Scikit-learn
    - Streamlit
    """)
    st.header("Project Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Records",
            "180,519"
        )
    with col2:
        st.metric(
           "Freight Model",
           "Random Forest Regressor" 
        )
    with col3:
        st.metric(
            "Risk Model",
            "Random Forest Classifier"
        )
    st.success(
        "Project Successfully Trained and ready for Predictions!"
    )

# Data insights page
elif page == "📊 Dataset Insights":
    st.header("📊 Dataset Insights")
    st.markdown("Explore the cleaned supply chain dataset and key business metrics.")

    # KPI SECTION
    total_records = len(df)
    avg_freight = round(
        df["Estimated_Freight_Cost"].mean(),2
    )
    risky_invoices = int(
        df["Risk_Flag"].sum()
    )
    avg_shipping_duration = round(
        df["Shipping_Duration"].mean(),2
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            "📦 Total Records",
            f"{total_records:,}"
        )
    with kpi2:
        st.metric(
            "🚚 Avg Freight Cost",
            avg_freight
        )
    with kpi3:
        st.metric(
            "⚠️ Risky Invoices",
            f"{risky_invoices:,}"
        )
    with kpi4:
        st.metric(
            "⏱ Avg Shipping Days",
            avg_shipping_duration
        )
    st.divider()

    # ROW 1
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Dataset Preview")
        st.dataframe(
            df.head(10),
            use_container_width=True
        )
    with col2:
        st.subheader("📈 Dataset Statistics")
        st.dataframe(
            df.describe(),
            use_container_width=True
        )
    st.divider()

    # ROW 2
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚠️ Risk Flag Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(
            data=df,
            x="Risk_Flag",
            ax=ax
        )
        ax.set_xlabel("Risk Flag")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    with col2:
        st.subheader("🚚 Freight Cost Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(
            df["Estimated_Freight_Cost"],
            bins=40,
            kde=True,
            ax=ax
        )
        ax.set_xlabel("Estimated Freight Cost")
        st.pyplot(fig)
    st.divider()

    # ROW 3
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⏱ Shipping Duration Analysis")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.boxplot(
            x=df["Shipping_Duration"],
            ax=ax
        )
        st.pyplot(fig)
    with col2:
        st.subheader("🧹 Missing Values Analysis")
        missing = (
            df.isnull()
            .sum()
        )
        missing = (
            missing[missing > 0]
            .sort_values(ascending=False)
        )
        if len(missing) > 0:
            st.dataframe(
                missing,
                use_container_width=True
            )
        else:
            st.success(
                "✅ No Missing Values Found"
            )
    st.divider()

    # ROW 4
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌍 Top Markets")
        market_data = (
            df["Market"]
            .value_counts()
            .head(10)
        )
        st.bar_chart(
            market_data
        )
    with col2:
        st.subheader("📍 Top Regions")
        region_data = (
            df["Order Region"]
            .value_counts()
            .head(10)
        )
        st.bar_chart(
            region_data
        )
    st.divider()

    # DOWNLOAD SECTION
    st.subheader("⬇ Download Dataset")
    csv = df.to_csv(
        index=False
    ).encode("utf-8")
    st.download_button(
        label="Download Clean Dataset",
        data=csv,
        file_name="cleaned_supply_chain_data.csv",
        mime="text/csv"
    )
# Freight cost prediction page
elif page == "🚚 Freight Cost Prediction":
    st.header("🚚 Freight Cost Prediction")
    st.markdown("""
    Predict the estimated freight cost 
    using order and shipping information
    """)
    st.subheader("Enter the order details: ")
    col1, col2 = st.columns(2)
    with col1:
        sales = st.number_input(
            "Sales Amount",
            min_value=0.0,
            value=1000.0
        )
        quantity = st.number_input(
            "Order Quantity",
            min_value=1,
            value=5
        )
        product_price = st.number_input(
            "Product price",
            min_value=0.0,
            value=100.0
        )
    with col2:
        shipping_duration = st.number_input(
            "Shipping Duration",
            min_value=0,
            value=4
        )
        late_delivery_risk = st.number_input(
            "Late Delivery Risk",
            min_value=0,
            max_value=1,
            value=0,
            step=1
        )
        scheduled_days = st.number_input(
            "Scheduled Shipping Days",
            min_value=0,
            value=3
        )
    predict_button = st.button("Predict Freight Cost")
    if predict_button:
        input_data = pd.DataFrame({
            "Sales":[sales],
            "Order Item Quantity":[quantity],
            "Order Item Product Price":[product_price],
            "Shipping_Duration":[shipping_duration],
            "Late_delivery_risk":[late_delivery_risk],
            "Days for shipment (scheduled)":[scheduled_days]
        })
        prediction = freight_model.predict(input_data)[0]
        st.success("Prediction Completed Successfully!")
        st.metric(
            label="🚚 Predicted Freight Cost",
            value=f"${prediction:,.2f}"
        )
        if prediction < low_threshold:
            st.info(
                "🟢 Low Freight Cost Order"
            )
        elif prediction < medium_threshold:
            st.warning(
                "🟡 Moderate Freight Cost Order"
            )
        else:
            st.error(
                "🔴 High Freight Cost Order"
        )
        st.subheader("Input Summary")
        st.dataframe(
            input_data,
            use_container_width=True
        )

# Risk Prediction Page
elif page == "⚠️ Risk Prediction":
    st.header("⚠️ Invoice Risk Prediction")
    st.markdown("""
    Predict whether an invoice is risky based 
    order and shipping information
    """)
    st.subheader("Enter invoice details: ")
    col1, col2 = st.columns(2)
    with col1:
        sales = st.number_input(
            "Sales Amount",
            min_value=0.0,
            value=1000.0,
            key="risk_sales"
        )
        quantity = st.number_input(
            "Order Quantity",
            min_value=1,
            value=5,
            key="risk_quantity"
        )
        product_price = st.number_input(
            "Product price",
            min_value=0.0,
            value=100.0,
            key="risk_price"
        )
        freight_cost = st.number_input(
            "Estimated Freight Cost",
            min_value=0.0,
            value=50.0
        )
    with col2:
        shipping_duration = st.number_input(
            "Shipping Duration",
            min_value=0,
            value=4,
            key="risk_shipping"
        )
        late_delivery_risk = st.number_input(
            "Late Delivery Risk",
            min_value=0,
            max_value=1,
            value=0,
            step=1,
            key="risk_late"
        )
        high_value_order = st.number_input(
            "High Value Order",
            min_value=0,
            max_value=1,
            value=0,
            step=1
        )
    predict_risk = st.button("Predict Invoice Risk")
    if predict_risk:
        input_data = pd.DataFrame({
            "Sales":[sales],
            "Order Item Quantity":[quantity],
            "Order Item Product Price":[product_price],
            "Shipping_Duration":[shipping_duration],
            "Late_delivery_risk":[late_delivery_risk],
            "Estimated_Freight_Cost":[freight_cost],
            "High_Value_Order":[high_value_order]
        })
        prediction = risk_model.predict(input_data)[0]
        st.success("Risk Analysis Completed Successfully!")
        if prediction == 0:
            st.success("🟢 Low Risk Invoice")
        else:
            st.error("🔴 High Risk Invoice")
        if prediction == 0:
            st.info(
                """
                Recommendation:\n
                • Process invoice normally\n
                • Standard audit review sufficient\n
                • Low operational risk
                """
            )
        else:
            st.warning(
                 """
                Recommendation:\n
                • Perform detailed invoice audit\n
                • Verify freight charges\n
                • Review delivery performance\n
                • Monitor vendor activity
                """
            )
        st.subheader("Input Summary")
        st.dataframe(
            input_data,
            use_container_width=True
        )

# Business analytics page
elif page == "📈 Business Analytics":
    st.header("📈 Business Analytics")
    st.markdown("""
    Analyze logistics performance and generate insights
    to optimize operations and reduce costs.
    """)
    total_sales = round(df["Sales"].sum(), 2)
    avg_freight = round(df["Estimated_Freight_Cost"].mean(), 2)
    total_risky = int(df["Risk_Flag"].sum())
    avg_shipping = round(df["Shipping_Duration"].mean(), 2)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            "💰 Total Sales",
            f"${total_sales:,.0f}"
        )
    with kpi2:
        st.metric(
            "🚚 Avg Freight",
            f"${avg_freight}"
        )
    with kpi3:
        st.metric(
           "⚠️ Risky Invoices",
            f"{total_risky:,}" 
        )
    with kpi4:
        st.metric(
            "⏱ Avg Shipping Days",
            avg_shipping
        )
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Invoice Risk Distribution")
        risk_counts = (df["Risk_Flag"].value_counts())
        st.bar_chart(risk_counts)
    with col2:
        st.subheader("Shipping Mode Usage")
        shipping_mode = (df["Shipping Mode"].value_counts())
        st.bar_chart(shipping_mode)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Markets")
        market_data = (df["Market"].value_counts().head(10))
        st.bar_chart(market_data)
    with col2:
        st.subheader("Top Regions")
        region_data = (df["Order Region"].value_counts().head(10))
        st.bar_chart(region_data)
    
    st.subheader("Sales by Market")
    sales_market = (df.groupby("Market")["Sales"].sum().sort_values(
        ascending=False
    ))
    st.bar_chart(sales_market)
    st.subheader("Delivery status Analysis")
    delivery_status = (df["Delivery Status"].value_counts())
    st.bar_chart(delivery_status)
    st.subheader("Freight Cost Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(
        df["Estimated_Freight_Cost"],
        bins=40,
        kde=True,
        ax=ax
    )
    st.pyplot(fig)
    st.divider()

    st.subheader(
        "Executive Summary"
    )
    st.info(
        f"""
        • Total Orders Processed: {len(df):,}\n
        • Risky Invoices Detected: {total_risky:,}\n
        • Average Freight Cost: ${avg_freight}\n
        • Average Shipping Duration: {avg_shipping} Days\n
        • Major Business Driver: Sales Value\n
        • Major Risk Driver: Late Delivery Risk
        """
    )