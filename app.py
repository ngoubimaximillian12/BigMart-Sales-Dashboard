import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load the trained XGBoost model
model = joblib.load("xgboost_model.pkl")

# ✅ Streamlit UI Layout
st.title("📊 BigMart Sales Prediction Dashboard")
st.write("Predict sales based on product and store attributes.")

# ✅ Sidebar for User Input
st.sidebar.header("Enter Product Details")
item_mrp = st.sidebar.number_input("Item MRP ($)", min_value=1.0, max_value=500.0, step=0.1)
item_weight = st.sidebar.number_input("Item Weight (kg)", min_value=1.0, max_value=50.0, step=0.1)
item_visibility = st.sidebar.slider("Item Visibility", 0.0, 1.0, 0.1)
outlet_type = st.sidebar.selectbox("Outlet Type", ["Supermarket", "Grocery Store"])
outlet_location_type = st.sidebar.selectbox("Outlet Location", ["Tier 1", "Tier 2", "Tier 3"])
outlet_size = st.sidebar.selectbox("Outlet Size", ["Small", "Medium", "Large"])
outlet_age = st.sidebar.slider("Outlet Age", 1, 50, 10)

# ✅ Convert User Input into Model Format
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

# ✅ Predict Sales
if st.sidebar.button("Predict Sales"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Sales: **${prediction:,.2f}**")

# ✅ Data Visualization - Feature Importance
st.subheader("Feature Importance")

# Extract feature importance from XGBoost
feature_importance = model.feature_importances_

# Get the correct feature names from the model
try:
    feature_names = model.get_booster().feature_names  # XGBoost stores feature names
except:
    feature_names = list(input_data.columns)  # Fallback to input column names

# Ensure feature importance matches feature names
if len(feature_names) == len(feature_importance):
    feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    # Plot Feature Importance
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="magma")
    plt.title("Feature Importance - XGBoost Model")
    st.pyplot(plt)
else:
    st.warning("⚠ Feature importance and feature names do not match. Skipping visualization.")

# ✅ Additional Insights
st.subheader("📌 Key Insights")
st.write("""
- **Item MRP** is the most important factor in predicting sales.
- **Supermarkets tend to have higher sales than grocery stores.**
- **Older outlets may see reduced sales performance.**
""")
