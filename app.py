import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------
# Title
# --------------------------------------------------------
st.title("Predictive Analysis — Paint Filling Volume & Size Classification")

# --------------------------------------------------------
# Load dataset
# --------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

df = load_data()
st.success("Dataset loaded successfully!")

# --------------------------------------------------------
# Encode categorical columns
# --------------------------------------------------------
cat_cols = df.select_dtypes(include=['object']).columns
encoders = {}

for col in cat_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col].astype(str))
    encoders[col] = enc

# --------------------------------------------------------
# Split data
# --------------------------------------------------------
X = df.drop("FILLING VOL", axis=1)
y = df["FILLING VOL"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------------------------
# Train Linear Regression model
# --------------------------------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# --------------------------------------------------------
# Input Section
# --------------------------------------------------------
st.subheader("Enter Product Details")

user_data = {}

for col in X.columns:
    # If feature was originally string → take text input then encode with saved encoder
    if col in encoders:
        val = st.text_input(col)
        if val != "":
            user_data[col] = encoders[col].transform([val])[0]
        else:
            user_data[col] = 0
    else:
        # numeric input
        val = st.number_input(col, value=0.0)
        user_data[col] = float(val)

# --------------------------------------------------------
# Predict Button
# --------------------------------------------------------
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([user_data])
        prediction = lr_model.predict(input_df)[0]

        st.subheader("Prediction")
        st.success(f"Predicted Filling Volume = {prediction:.3f} Litres")

    except Exception as e:
        st.error(f"Error: {e}")

st.info("App ready! Fully functional.")
