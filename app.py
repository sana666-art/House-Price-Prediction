import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction App")
st.write("Predict house prices using **Linear Regression**")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    return data.frame

df = load_data()

# -----------------------------
# Prepare Data
# -----------------------------
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🏗️ House Features")

MedInc = st.sidebar.slider(
    "Median Income (10k USD)", 
    float(X["MedInc"].min()), 
    float(X["MedInc"].max()), 
    float(X["MedInc"].mean())
)

HouseAge = st.sidebar.slider(
    "House Age (years)", 
    float(X["HouseAge"].min()), 
    float(X["HouseAge"].max()), 
    float(X["HouseAge"].mean())
)

AveRooms = st.sidebar.slider(
    "Average Rooms", 
    float(X["AveRooms"].min()), 
    float(X["AveRooms"].max()), 
    float(X["AveRooms"].mean())
)

AveBedrms = st.sidebar.slider(
    "Average Bedrooms", 
    float(X["AveBedrms"].min()), 
    float(X["AveBedrms"].max()), 
    float(X["AveBedrms"].mean())
)

Population = st.sidebar.slider(
    "Population", 
    float(X["Population"].min()), 
    float(X["Population"].max()), 
    float(X["Population"].mean())
)

AveOccup = st.sidebar.slider(
    "Average Occupancy", 
    float(X["AveOccup"].min()), 
    float(X["AveOccup"].max()), 
    float(X["AveOccup"].mean())
)

Latitude = st.sidebar.slider(
    "Latitude", 
    float(X["Latitude"].min()), 
    float(X["Latitude"].max()), 
    float(X["Latitude"].mean())
)

Longitude = st.sidebar.slider(
    "Longitude", 
    float(X["Longitude"].min()), 
    float(X["Longitude"].max()), 
    float(X["Longitude"].mean())
)

# -----------------------------
# Prediction
# -----------------------------
input_data = np.array([
    MedInc, HouseAge, AveRooms, AveBedrms,
    Population, AveOccup, Latitude, Longitude
]).reshape(1, -1)

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

# -----------------------------
# Output
# -----------------------------
st.subheader("📈 Predicted House Price")

st.success(
    f"Estimated Price: **${prediction * 100000:,.0f} USD**"
)

# -----------------------------
# Model Performance
# -----------------------------
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)

st.caption(f"Model R² Score: {r2:.2f}")