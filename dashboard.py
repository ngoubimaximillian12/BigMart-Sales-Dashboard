import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ✅ Load the trained XGBoost model
model = joblib.load("xgboost_model.pkl")

# ✅ Load Dataset for Visualizations
dataset_path = "/Users/ngoubimaximilliandiamgha/.cache/kagglehub/datasets/shivan118/big-mart-sales-prediction-datasets/versions/1/train.csv"
df = pd.read_csv(dataset_path)

# ✅ Dashboard UI Layout
st.set_page_config(page_title="BigMart Sales Dashboard", layout="wide")

# 🔹 Sidebar - User Input for Predictions
st.sidebar.header("📌 Enter Product Details")
item_mrp = st.sidebar.number_input("Item MRP ($)", min_value=1.0, max_value=500.0, step=0.1)
item_weight = st.sidebar.number_input("Item Weight (kg)", min_value=1.0, max_value=50.0, step=0.1)
item_visibility = st.sidebar.slider("Item Visibility", 0.0, 1.0, 0.1)
outlet_type = st.sidebar.selectbox("Outlet Type", ["Supermarket", "Grocery Store"])
outlet_location_type = st.sidebar.selectbox("Outlet Location", ["Tier 1", "Tier 2", "Tier 3"])
outlet_size = st.sidebar.selectbox("Outlet Size", ["Small", "Medium", "Large"])
outlet_age = st.sidebar.slider("Outlet Age", 1, 50, 10)

# 🔹 Convert User Input to Model Format
input_data = pd.DataFrame({
    "Item_MRP": [item_mrp],
    "Item_Weight": [item_weight],
    "Item_Visibility": [item_visibility],
    "Outlet_Type": [1 if outlet_type == "Supermarket" else 0],
    "Outlet_Location_Type_Tier 2": [1 if outlet_location_type == "Tier 2" else 0],
    "Outlet_Location_Type_Tier 3": [1 if outlet_location_type == "Tier 3" else 0],
    "Outlet_Size_Medium": [1 if outlet_size == "Medium" else 0],
    "Outlet_Size_Large": [1 if outlet_size == "Large" else 0],
    "Outlet_Age": [outlet_age]
})

# 🔹 Predict Sales
if st.sidebar.button("Predict Sales"):
    prediction = model.predict(input_data)[0]
    st.sidebar.success(f"Predicted Sales: **${prediction:,.2f}**")

# 🔹 Page Title
st.title("📊 BigMart Sales Dashboard")
st.markdown("### Explore historical sales data and predict future sales trends.")

# ✅ **1️⃣ Sales Distribution by Outlet Type**
st.subheader("🛒 Sales Distribution Across Different Outlet Types")
fig = px.box(df, x="Outlet_Type", y="Item_Outlet_Sales", color="Outlet_Type", title="Sales by Outlet Type")
st.plotly_chart(fig, use_container_width=True)

# ✅ **2️⃣ Item MRP vs Sales**
st.subheader("💰 Price vs Sales Analysis")
fig = px.scatter(df, x="Item_MRP", y="Item_Outlet_Sales", color="Outlet_Type", title="MRP vs Sales")
st.plotly_chart(fig, use_container_width=True)

# ✅ **3️⃣ Feature Importance Analysis**
st.subheader("🎯 Feature Importance in Sales Prediction")

# Extract feature importance from XGBoost
feature_importance = model.feature_importances_
try:
    feature_names = model.get_booster().feature_names
except:
    feature_names = list(input_data.columns)

# Ensure feature importance matches feature names
if len(feature_names) == len(feature_importance):
    feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    fig = px.bar(feature_importance_df, x="Importance", y="Feature", orientation="h", title="Feature Importance - XGBoost")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠ Feature importance visualization skipped due to mismatch.")

# ✅ **4️⃣ Outlet Sales Over Time**
st.subheader("📈 Outlet Sales Trend Over Years")
df["Outlet_Age"] = 2025 - df["Outlet_Establishment_Year"]
outlet_sales = df.groupby("Outlet_Age")["Item_Outlet_Sales"].sum().reset_index()
fig = px.line(outlet_sales, x="Outlet_Age", y="Item_Outlet_Sales", title="Sales Trend by Outlet Age")
st.plotly_chart(fig, use_container_width=True)

# ✅ **5️⃣ Additional Insights**
st.subheader("📌 Key Insights")
st.write("""
- **Item MRP** has a strong positive correlation with sales.
- **Supermarkets consistently outperform grocery stores.**
- **Older outlets tend to have reduced sales performance.**
""")
