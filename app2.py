import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction App")
st.write("End-to-end ML app using Pipeline, EDA & Model Analysis")

# ---------------------------------
# Load Dataset
# ---------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/housing.csv")

df = load_data()

# ---------------------------------
# Split Features & Target
# ---------------------------------
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# ---------------------------------
# Preprocessing Pipelines
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
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# ---------------------------------
# Train / Test Split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------
# Model Save / Load
# ---------------------------------
MODEL_PATH = "model/house_price_pipeline.pkl"
os.makedirs("model", exist_ok=True)

if os.path.exists(MODEL_PATH):
    pipeline = joblib.load(MODEL_PATH)
else:
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)

# ---------------------------------
# Tabs
# ---------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Prediction", "📊 Dataset Overview", "📈 EDA Visuals", "📉 Model Analysis"]
)

# ======================================================
# 🔮 TAB 1: PREDICTION
# ======================================================
with tab1:
    st.subheader("🔮 Predict House Price")

    st.sidebar.header("🏠 Input Features")

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

    prediction = pipeline.predict(input_data)[0]

    st.success(f"💰 Estimated House Price: **${prediction:,.0f}**")

# ======================================================
# 📊 TAB 2: DATASET OVERVIEW
# ======================================================
with tab2:
    st.subheader("📊 Dataset Overview")

    st.write("**Dataset Shape:**", df.shape)

    st.write("### Column Names")
    st.write(df.columns.tolist())

    st.write("### Missing Values")
    st.write(df.isnull().sum())

    st.write("### Sample Records")
    st.dataframe(df.head())

# ======================================================
# 📈 TAB 3: EDA VISUALS
# ======================================================
with tab3:
    st.subheader("📈 Exploratory Data Analysis")

    st.write("### Correlation Heatmap (Numerical Features)")
    corr = df.select_dtypes(include=["int64", "float64"]).corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.write("### Median Income vs House Price")
    fig_scatter = px.scatter(
        df,
        x="median_income",
        y="median_house_value",
        color="ocean_proximity",
        title="Median Income vs House Price"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ======================================================
# 📉 TAB 4: MODEL ANALYSIS
# ======================================================
with tab4:
    st.subheader("📉 Model Performance & Residual Analysis")

    y_test_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_test_pred)

    st.metric("R² Score", round(r2, 3))

    residuals = y_test - y_test_pred

    st.write("### Residual Distribution")
    fig_resid = px.histogram(
        residuals,
        nbins=50,
        title="Residual Distribution"
    )
    st.plotly_chart(fig_resid, use_container_width=True)

    st.write("### Actual vs Predicted Prices")
    fig_actual = px.scatter(
        x=y_test,
        y=y_test_pred,
        labels={"x": "Actual Prices", "y": "Predicted Prices"},
        title="Actual vs Predicted Prices"
    )
    st.plotly_chart(fig_actual, use_container_width=True)