# ✅ Import Required Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ✅ Step 2: Hypothesis Generation
hypotheses = {
    "Higher MRP leads to higher sales": "MRP vs Sales",
    "Supermarkets have higher sales than Grocery Stores": "Outlet Type vs Sales",
    "Older outlets may have lower sales": "Outlet Age vs Sales",
    "Items with low visibility may have lower sales": "Item Visibility vs Sales"
}
print("\nHypothesis Generation")
for hypothesis, test in hypotheses.items():
    print(f"- {hypothesis} ({test})")

# ✅ Step 3: Loading Data
dataset_path = "/Users/ngoubimaximilliandiamgha/.cache/kagglehub/datasets/shivan118/big-mart-sales-prediction-datasets/versions/1/train.csv"

if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
    print("\nDataset Loaded Successfully!")
else:
    raise FileNotFoundError(f"File not found: {dataset_path}. Check the path!")

# ✅ Step 4: Data Structure and Content
print("\nDataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:\n", df.isnull().sum())
print("\nDataset Description:\n", df.describe())

# ✅ Step 6: Univariate Analysis
plt.figure(figsize=(8, 6))
sns.histplot(df['Item_Outlet_Sales'], bins=50, kde=True)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

# ✅ Step 7: Bivariate Analysis
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Item_MRP', y='Item_Outlet_Sales', data=df)
plt.title("MRP vs Sales")
plt.xlabel("MRP")
plt.ylabel("Sales")
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Outlet_Type', y='Item_Outlet_Sales', data=df)
plt.title("Sales by Outlet Type")
plt.xticks(rotation=45)
plt.show()

# ✅ Step 8: Missing Value Treatment
if df['Item_Weight'].isnull().sum() > 0:
    imputer = SimpleImputer(strategy='median')
    df['Item_Weight'] = imputer.fit_transform(df[['Item_Weight']])

df.loc[:, 'Outlet_Size'] = df['Outlet_Size'].fillna("Unknown")

# ✅ Step 9: Feature Engineering
df['Item_Age'] = 2023 - df['Outlet_Establishment_Year']
df['Sales_Per_Unit_Visibility'] = df['Item_Outlet_Sales'] / (df['Item_Visibility'] + 1e-6)

# ✅ Step 10-12: Encoding Categorical Variables
le = LabelEncoder()
df['Outlet_Identifier'] = le.fit_transform(df['Outlet_Identifier'])
df['Outlet_Type'] = le.fit_transform(df['Outlet_Type'])

df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})
df['Item_Fat_Content'] = le.fit_transform(df['Item_Fat_Content'])

df = pd.get_dummies(df, columns=['Item_Type', 'Outlet_Location_Type', 'Outlet_Size'], drop_first=True)

# ✅ Step 13: Preprocessing Data
X = df.drop(['Item_Outlet_Sales', 'Item_Identifier', 'Outlet_Establishment_Year'], axis=1)
y = df['Item_Outlet_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Step 14-18: Modeling

# 1) Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\nLinear Regression R2 Score:", r2_score(y_test, y_pred_lr))

# 2) Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
print("\nRidge Regression R2 Score:", r2_score(y_test, y_pred_ridge))

# 3) Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest R2 Score:", r2_score(y_test, y_pred_rf))

# 4) XGBoost with Hyperparameter Tuning
xgb = XGBRegressor(random_state=42)
xgb_params = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}
grid_search_xgb = GridSearchCV(xgb, xgb_params, cv=3, scoring='r2', n_jobs=-1)
grid_search_xgb.fit(X_train, y_train)

best_xgb = grid_search_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
print("\nXGBoost R2 Score (After Tuning):", r2_score(y_test, y_pred_xgb))

# ✅ Step 19: Feature Importance Graph (Fixed Warning)
feature_importance = best_xgb.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_importance_df, palette='viridis', legend=False)
plt.title('Feature Importance - XGBoost')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# ✅ Step 20: Saving Models
joblib.dump(lr, "linear_regression_model.pkl")
joblib.dump(ridge, "ridge_regression_model.pkl")
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(best_xgb, "xgboost_model.pkl")

print("\nFinal models saved successfully!")
