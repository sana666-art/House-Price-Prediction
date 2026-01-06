import streamlit as st
import numpy as np
import pandas as pd

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
    df = pd.read_csv("data/housing.csv")

    # -----------------------------
    # data Preprocessing
    # -----------------------------

    # Step 1:
    # Fill numeric missing values with median
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Step 2:
    # Fill missing categorical values
    df['ocean_proximity'].fillna('UNKNOWN', inplace=True)


    # Step 3:
    # Encode categorical column
    df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

    return df

df = load_data()

# -----------------------------
# Prepare Data
# -----------------------------
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
st.sidebar.header("🏠 House Features")

longitude = st.sidebar.slider(
    "Longitude",
    float(X["longitude"].min()),
    float(X["longitude"].max()),
    float(X["longitude"].mean())
)

latitude = st.sidebar.slider(
    "Latitude",
    float(X["latitude"].min()),
    float(X["latitude"].max()),
    float(X["latitude"].mean())
)

housing_median_age = st.sidebar.slider(
    "Housing Median Age",
    float(X["housing_median_age"].min()),
    float(X["housing_median_age"].max()),
    float(X["housing_median_age"].mean())
)

total_rooms = st.sidebar.slider(
    "Total Rooms",
    float(X["total_rooms"].min()),
    float(X["total_rooms"].max()),
    float(X["total_rooms"].mean())
)

total_bedrooms = st.sidebar.slider(
    "Total Bedrooms",
    float(X["total_bedrooms"].min()),
    float(X["total_bedrooms"].max()),
    float(X["total_bedrooms"].mean())
)

population = st.sidebar.slider(
    "Population",
    float(X["population"].min()),
    float(X["population"].max()),
    float(X["population"].mean())
)

households = st.sidebar.slider(
    "Households",
    float(X["households"].min()),
    float(X["households"].max()),
    float(X["households"].mean())
)

median_income = st.sidebar.slider(
    "Median Income",
    float(X["median_income"].min()),
    float(X["median_income"].max()),
    float(X["median_income"].mean())
)

# -----------------------------
# Prediction
# -----------------------------
input_data = np.array([
    longitude, latitude, housing_median_age, total_rooms,
    total_bedrooms, population, households, median_income
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