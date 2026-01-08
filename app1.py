import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction (Pipeline Version)")
st.write("ML pipeline with preprocessing + Linear Regression")

# ---------------------------------
# Load data
# ---------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/housing.csv")

df = load_data()

# ---------------------------------
# Split features & target
# ---------------------------------
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Identify column types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# ---------------------------------
# Preprocessing pipelines
# ---------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ---------------------------------
# Full ML Pipeline
# ---------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# ---------------------------------
# Train / Test split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------
# Sidebar inputs
# ---------------------------------
st.sidebar.header("🏠 House Features")

longitude = st.sidebar.slider("Longitude", float(df.longitude.min()), float(df.longitude.max()), float(df.longitude.mean()))
latitude = st.sidebar.slider("Latitude", float(df.latitude.min()), float(df.latitude.max()), float(df.latitude.mean()))
housing_median_age = st.sidebar.slider("Housing Median Age", int(df.housing_median_age.min()), int(df.housing_median_age.max()), int(df.housing_median_age.mean()))
total_rooms = st.sidebar.slider("Total Rooms", int(df.total_rooms.min()), int(df.total_rooms.max()), int(df.total_rooms.mean()))
total_bedrooms = st.sidebar.slider("Total Bedrooms", int(df.total_bedrooms.min()), int(df.total_bedrooms.max()), int(df.total_bedrooms.mean()))
population = st.sidebar.slider("Population", int(df.population.min()), int(df.population.max()), int(df.population.mean()))
households = st.sidebar.slider("Households", int(df.households.min()), int(df.households.max()), int(df.households.mean()))
median_income = st.sidebar.slider("Median Income", float(df.median_income.min()), float(df.median_income.max()), float(df.median_income.mean()))

ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    df["ocean_proximity"].unique()
)

# ---------------------------------
# Create input DataFrame
# ---------------------------------
input_data = pd.DataFrame([{
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity
}])

# ---------------------------------
# Prediction
# ---------------------------------
prediction = model.predict(input_data)[0]

st.subheader("📈 Predicted House Price")
st.success(f"${prediction:,.0f}")

# ---------------------------------
# Model performance
# ---------------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

st.caption(f"Model R² Score: {r2:.2f}")