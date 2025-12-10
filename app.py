import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Paint Volume Prediction", layout="wide")

st.title("Predictive Analysis — Paint Filling Volume & Size Classification")

# -------------------------
# Load dataset from GitHub
# -------------------------
GITHUB_DATA_URL = "https://raw.githubusercontent.com/Yashvishwakarma19/paint-volume-prediction-ml/main/PredactiveDataSet.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(GITHUB_DATA_URL)

df = load_data()

st.success("Dataset loaded successfully from GitHub!")
st.write(df.head())

# -------------------------
# Preprocessing
# -------------------------

# Remove missing values
df = df.dropna()

# Select numeric columns for regression
numeric_df = df.select_dtypes(include=[np.number])

X = numeric_df.drop(columns=["FILLING VOL"])
y = numeric_df["FILLING VOL"]

# -------------------------
# Linear Regression Model
# -------------------------
lin_model = LinearRegression()
lin_model.fit(X, y)

# -------------------------
# Polynomial Regression Model
# -------------------------
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# -------------------------
# Decision Tree Classification Model
# -------------------------
df["VOLUME_CATEGORY"] = df["FILLING VOL"].apply(
    lambda x: "SMALL" if x < 2 else ("MEDIUM" if x < 5 else "LARGE")
)

clf = DecisionTreeClassifier()
clf.fit(X, df["VOLUME_CATEGORY"])

# -------------------------
# User Input Section
# -------------------------
st.header("Enter product details to predict")

input_data = {}

for col in X.columns:
    input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))

input_df = pd.DataFrame([input_data])

# -------------------------
# Predictions
# -------------------------
st.subheader("Predictions")

lin_pred = lin_model.predict(input_df)[0]
poly_pred = poly_model.predict(poly.transform(input_df))[0]
tree_pred = clf.predict(input_df)[0]

st.write(f"**Linear Regression predicted FILLING VOL = {lin_pred:.4f}**")
st.write(f"**Polynomial Regression predicted FILLING VOL = {poly_pred:.4f}**")
st.write(f"**Decision Tree predicted VOLUME_CATEGORY = {tree_pred}**")

st.success("App ready!")

st.write("---")
st.write("This app loads the dataset automatically from GitHub — no upload required.")
