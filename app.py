import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Paint Prediction", layout="wide")

st.title("Predictive Analysis — Paint Filling Volume & Size Classification")

# -------------------------
# Load dataset from GitHub
# -------------------------
URL = "https://raw.githubusercontent.com/Yashvishwakarma19/paint-volume-prediction-ml/main/PredactiveDataSet.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(URL)

df = load_data()
st.success("Dataset loaded successfully!")

# -------------------------
# CLEANING
# -------------------------
df = df.dropna()

# Create volume category
df["VOLUME_CATEGORY"] = df["FILLING VOL"].apply(
    lambda x: "SMALL" if x < 2 else ("MEDIUM" if x < 5 else "LARGE")
)

# -------------------------
# ENCODING CATEGORICAL COLUMNS
# -------------------------
encode_cols = df.select_dtypes(include=["object"]).columns
encoders = {}

for col in encode_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col].astype(str))
    encoders[col] = enc

# -------------------------
# FEATURES & TARGETS
# -------------------------
X = df.drop(columns=["FILLING VOL", "VOLUME_CATEGORY"])
y = df["FILLING VOL"]

# -------------------------
# MODELS
# -------------------------
lin = LinearRegression()
lin.fit(X, y)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

clf = DecisionTreeClassifier()
clf.fit(X, df["VOLUME_CATEGORY"])

# -------------------------
# USER INPUT FORM
# -------------------------
st.header("Enter Product Details")

user_data = {}

for col in X.columns:

    # If categorical → dropdown with unique original labels
    if col in encode_cols:
        original_values = encoders[col].classes_
        user_value = st.selectbox(f"{col}", original_values)
        user_data[col] = encoders[col].transform([user_value])[0]

    else:
        # numeric → use median default
        default_val = float(df[col].median())
        user_value = st.number_input(f"{col}", value=default_val)
        user_data[col] = user_value

input_df = pd.DataFrame([user_data])

# -------------------------
# PREDICTIONS
# -------------------------
st.subheader("Predictions")

lin_pred = lin.predict(input_df)[0]
poly_pred = poly_model.predict(poly.transform(input_df))[0]
tree_pred = clf.predict(input_df)[0]

# Decode category back to text (SMALL / MEDIUM / LARGE)
tree_pred_label = encoders["VOLUME_CATEGORY"].inverse_transform([tree_pred])[0]

st.write(f"**Linear Regression FILLING VOL = {lin_pred:.3f}**")
st.write(f"**Polynomial Regression FILLING VOL = {poly_pred:.3f}**")
st.write(f"**Decision Tree VOLUME CATEGORY = {tree_pred_label}**")

st.success("App ready! Fully functional.")
