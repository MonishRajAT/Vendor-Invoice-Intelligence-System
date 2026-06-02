# 📦 Vendor Invoice Intelligence System

## 🚀 Overview

The Vendor Invoice Intelligence System is an analytics platform I developed to help companies get a handle on their logistics costs. The core goal is to use machine learning to predict freight expenses and flag high-risk invoices before they become a financial headache.

By combining data analytics, ML, and business intelligence, the system transforms raw supply chain data into a tool for reducing auditing effort, monitoring operational risks, and making smarter business decisions.

---

## 🎯 The Business Problem

In the world of logistics, companies are often flying blind. They deal with unexpected freight spikes, delivery delays, and an auditing process that is slow, manual, and prone to error.

I designed this system to solve those specific pain points by delivering three main things:

1. Accurate Freight Cost Predictions (so budgets aren't blown).
2. Automated Invoice Risk Detection (to find the "red flags" instantly).
3. A Business Analytics Dashboard (to give leadership a clear view of performance).

---

## 📊 The Data

I used the DataCo Smart Supply Chain Dataset, which is substantial in scope:

- Scale: 180,519 records with 53 different features.
- What's inside: I tracked a wide range of attributes, including Sales, Product Price, Order Quantity, Shipping Duration, Delivery Status, Shipping Mode, Market, Region, and Profit information.

---

## ⚙️ Project Workflow

The project follows a logical end-to-end pipeline:

Raw Dataset → Data Cleaning → Feature Engineering → EDA → Freight Cost Model → Risk Classification Model → Model Explainability → Streamlit Dashboard → Business Insights

---

## 🧹 Data Preprocessing

Data is never clean when it arrives, so I put it through a rigorous preprocessing phase.

### Data Cleaning

- Stripped out customer-sensitive attributes to maintain privacy.
- Cleaned up missing values and removed duplicate records.
- Standardized all date columns into a proper datetime format.

### Feature Engineering

To make the models actually "understand" the business, I created several custom features:

- Shipping_Duration
- Order_Value
- Profit_Margin
- Estimated_Freight_Cost
- High_Value_Order
- Risk_Flag

---

## 📈 Exploratory Data Analysis (EDA)

During the EDA phase, I uncovered a few key drivers that shape the logistics process.

### What drives Freight Costs?

- Sales had the strongest correlation with the final cost.
- Product Price and Shipping Duration also played significant roles in driving up expenses.

### What drives Invoice Risk?

- The biggest predictor of risk was Late Delivery Risk.
- High-Value Orders and longer Shipping Durations also consistently increased the likelihood of a "risk flag."

---

## 🤖 Machine Learning Implementation

### 1. Freight Cost Prediction (Regression)

The goal here was to predict the Estimated_Freight_Cost.

- Algorithms Tested: I compared Linear Regression and Random Forest Regressor.
- The Winner: Random Forest Regressor provided the best performance.
- How I measured success: I used MAE, RMSE, and the R² Score to validate accuracy.

### 2. Invoice Risk Prediction (Classification)

The goal here was to predict the Risk_Flag.

- Algorithms Tested: I compared Logistic Regression and Random Forest Classifier.
- The Winner: Random Forest Classifier was the most reliable.
- How I measured success: I focused on Accuracy, Precision, Recall, and the F1 Score.

---

## 🔍 Model Explainability

I didn't want these to be "black box" models, so I ran a feature importance analysis to see exactly what the AI was looking at:

### Freight Cost Drivers

- Sales
- Product Price
- Shipping Duration

### Invoice Risk Drivers

- Late Delivery Risk
- High Value Orders
- Freight Cost

---

## 📊 Streamlit Dashboard Features

I wrapped the entire system in a Streamlit app to make it accessible to non-technical users.

### 🏠 Home

- Project Overview
- System Summary
- KPI Cards

### 📊 Dataset Insights

- Dataset Preview
- Statistical Summary
- Risk Distribution Analysis
- Freight Cost Analysis
- Shipping Duration Analysis
- Market and Region Insights

### 🚚 Freight Cost Prediction

- Interactive User Input Form
- Real-Time Cost Prediction
- Freight Cost Categorization

### ⚠️ Invoice Risk Prediction

- Risk Detection
- Business Recommendations
- Invoice Analysis

### 📈 Business Analytics

- Executive Dashboard
- Risk Monitoring
- Market Analysis
- Delivery Performance Analysis
- Freight Cost Insights

---

## 🛠️ Technology Stack

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-Learn

### Deployment

- Streamlit

### Model Persistence

- Joblib

---

## 📂 Project Structure

```text
Vendor_Invoice_Intelligence_System/

├── app/
│   └── streamlit_app.py

├── data/
│   ├── DataCoSupplyChainDataset.csv
│   └── cleaned_supply_chain_data.csv

├── models/
│   ├── freight_model.pkl
│   └── risk_model.pkl

├── reports/
│   ├── feature_importance.csv
│   └── freight_feature_importance.csv

├── notebooks/

├── requirements.txt

└── README.md
```

---

## 💼 Business Value

This system isn't just a technical exercise; it provides real-world value by:

- Lowering the cost of auditing through automation.
- Predicting logistics expenses to help with budgeting.
- Identifying high-risk vendors and invoices early.
- Giving leadership a data-driven way to monitor supply chain health.

---

## 🔮 Future Enhancements

There's always room to grow. My roadmap for v2 includes:

- Integrating real-time freight cost APIs.
- Developing a Vendor Performance Scoring System.
- Moving the system to a cloud deployment (AWS/Azure).
- Implementing SHAP for deeper model explainability.
- Building more interactive Plotly dashboards.
- Automated reporting and alert systems.

---

## 👨‍💻 Author

**Monish Raj A T**

Undergrad pursuing B.Tech in Artificial Intelligence & Machine Learning
