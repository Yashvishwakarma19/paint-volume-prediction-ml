# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import DecisionBoundaryDisplay

st.set_page_config(page_title="Paint Fill Predictor", layout="wide")

st.title("Predictive Analysis — Paint Filling Volume & Size Classification")
st.markdown("Upload your dataset (Excel) or use the default file path.")

# ---- Upload or load file ----
uploaded_file = st.file_uploader("Upload Excel file (same format as your dataset)", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif file_name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)

    else:
        st.error("Please upload a CSV or Excel file.")
        st.stop()

    st.success("File uploaded successfully!")
    st.dataframe(df)
else:
    st.warning("Please upload a dataset to continue.")
    st.stop()

st.subheader("Preview data (first 5 rows)")
st.dataframe(df.head())

# ---- Preprocessing functions (same logic you used) ----
def convert_to_liters(x):
    try:
        x = str(x).strip().upper()
        if "ML" in x:
            value = float(x.replace("ML", "").strip())
            return value / 1000
        elif "LTR" in x:
            value = float(x.replace("LTR", "").strip())
            return float(x.replace("LTR", "").strip())
        elif "L" in x:
            value = float(x.replace("L", "").strip())
            return float(x.replace("L", "").strip())
        else:
            return None
    except:
        return None

def map_fms(x):
    try:
        x = str(x).strip().upper()
        if x == "S":
            return "Standard"
        elif x == "M":
            return "Medium"
        elif x == "F":
            return "Fine"
        else:
            return None
    except:
        return None

@st.cache_data
def preprocess(df):
    df = df.copy()
    if 'PACK SIZE' in df.columns:
        df['PACK_SIZE_LITERS'] = df['PACK SIZE'].apply(convert_to_liters)
    else:
        st.error("PACK SIZE column not found in dataset.")
    if 'FMS' in df.columns:
        df['FMS_MAPPED'] = df['FMS'].apply(map_fms)
        df.drop(columns=['FMS'], inplace=True, errors='ignore')
    df.drop(columns=['Description'], inplace=True, errors='ignore')
    df['VOLUME_CATEGORY'] = df['PACK_SIZE_LITERS'].apply(lambda x: "SMALL" if x <= 1 else ("MEDIUM" if x <= 5 else "LARGE"))
    df['RB_ENCODED'] = df['R/B'].map({'Retail':0, 'Bulk':1})
    return df

df = preprocess(df)
st.write("Columns after preprocessing:", list(df.columns))

# ---- Prepare regression & classification datasets ----
# Regression X (use get_dummies as earlier)
X_reg = df.drop(['FILLING VOL', 'SKU CODE'], axis=1, errors='ignore')
X_reg = pd.get_dummies(X_reg, drop_first=True).fillna(0)
y_reg = df['FILLING VOL']

# Classification X (drop VOLUME_CATEGORY)
X_cls = df.drop(['VOLUME_CATEGORY', 'SKU CODE'], axis=1, errors='ignore')
X_cls = pd.get_dummies(X_cls, drop_first=True).fillna(0)
y_cls = df['VOLUME_CATEGORY']

# Make sure columns align (in case get_dummies produced different shapes)
X_reg, X_cls = X_reg.align(X_cls, join='outer', axis=1, fill_value=0)

# ---- Train / Test split (consistent) ----
test_size = st.sidebar.slider("Test size (fraction)", 0.1, 0.4, 0.2)
random_state = 42

# For regression, scale features
scaler = MinMaxScaler()
X_reg_scaled = pd.DataFrame(scaler.fit_transform(X_reg), columns=X_reg.columns)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg_scaled, y_reg, test_size=test_size, random_state=random_state)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cls, y_cls, test_size=test_size, random_state=random_state)

# ---- Train models button ----
if st.button("Train all models"):
    st.info("Training models... this may take a few seconds")

    # Linear Regression
    lr = LinearRegression()
    lr.fit(Xr_train, yr_train)
    yr_pred = lr.predict(Xr_test)
    r2 = r2_score(yr_test, yr_pred)
    mse = mean_squared_error(yr_test, yr_pred)
    rmse = np.sqrt(mse)

    # Polynomial (degree=2)
    poly = PolynomialFeatures(degree=2)
    Xr_train_poly = poly.fit_transform(Xr_train)
    Xr_test_poly = poly.transform(Xr_test)
    lr_poly = LinearRegression()
    lr_poly.fit(Xr_train_poly, yr_train)
    yr_pred_poly = lr_poly.predict(Xr_test_poly)
    r2_poly = r2_score(yr_test, yr_pred_poly)
    mse_poly = mean_squared_error(yr_test, yr_pred_poly)
    rmse_poly = np.sqrt(mse_poly)

    # KNN classifier (for VOLUME_CATEGORY)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Xc_train, yc_train)
    yc_pred_knn = knn.predict(Xc_test)
    acc_knn = accuracy_score(yc_test, yc_pred_knn)

    # Decision Tree classifier
    dt = DecisionTreeClassifier(criterion='entropy', random_state=random_state)
    dt.fit(Xc_train, yc_train)
    yc_pred_dt = dt.predict(Xc_test)
    acc_dt = accuracy_score(yc_test, yc_pred_dt)

    # SVM classifier (on same Xc_train; SVC requires numeric labels for plotting later)
    svm = SVC(kernel='rbf')
    svm.fit(Xc_train, yc_train)
    yc_pred_svm = svm.predict(Xc_test)
    acc_svm = accuracy_score(yc_test, yc_pred_svm)

    # Save models/results in session state
    st.session_state['models'] = {
        'lr': lr, 'lr_poly': lr_poly, 'poly': poly,
        'knn': knn, 'dt': dt, 'svm': svm,
        'yr_pred': yr_pred, 'yr_pred_poly': yr_pred_poly,
        'yc_pred_knn': yc_pred_knn, 'yc_pred_dt': yc_pred_dt, 'yc_pred_svm': yc_pred_svm,
        'r2': r2, 'mse': mse, 'rmse': rmse,
        'r2_poly': r2_poly, 'mse_poly': mse_poly, 'rmse_poly': rmse_poly,
        'acc_knn': acc_knn, 'acc_dt': acc_dt, 'acc_svm': acc_svm
    }

    st.success("Training completed. Scroll down for results.")

# ---- Show results if trained ----
if 'models' in st.session_state:
    m = st.session_state['models']

    st.header("Regression Results")
    st.write("Linear Regression — R²: {:.4f}, RMSE: {:.4f}".format(m['r2'], m['rmse']))
    st.write("Polynomial Regression (deg2) — R²: {:.4f}, RMSE: {:.4f}".format(m['r2_poly'], m['rmse_poly']))

    st.header("Classification Results")
    st.write(f"KNN Accuracy: {m['acc_knn']:.4f}")
    st.write(f"Decision Tree Accuracy: {m['acc_dt']:.4f}")
    st.write(f"SVM Accuracy: {m['acc_svm']:.4f}")

    st.subheader("Confusion Matrices")
    st.write("KNN")
    st.write(confusion_matrix(yc_test, m['yc_pred_knn']))
    st.write("Decision Tree")
    st.write(confusion_matrix(yc_test, m['yc_pred_dt']))
    st.write("SVM")
    st.write(confusion_matrix(yc_test, m['yc_pred_svm']))

    # Decision Tree plot
    st.subheader("Decision Tree Visualization")
    fig = plt.figure(figsize=(12,6))
    plot_tree(m['dt'], feature_names=Xc_train.columns, class_names=np.unique(yc_train).astype(str), filled=True, rounded=True, fontsize=8)
    st.pyplot(fig)

    # SVM decision boundary (2 features)
    if Xc_train.shape[1] >= 2:
        st.subheader("SVM Decision Boundary (first two features)")
        X_vis = Xc_train.iloc[:, :2].values
        y_vis = yc_train.values
        y_vis_codes = pd.Categorical(y_vis).codes
        svm_vis = SVC(kernel='rbf')
        svm_vis.fit(X_vis, y_vis_codes)
        fig2 = plt.figure(figsize=(8,5))
        DecisionBoundaryDisplay.from_estimator(svm_vis, X_vis, response_method='predict', cmap='Blues', alpha=0.6)
        plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis_codes, edgecolors='k', s=40)
        st.pyplot(fig2)

    # Allow prediction on a selected row
    st.subheader("Predict on a dataset row")
    idx = st.number_input("Enter row index (0 - {})".format(len(df)-1), min_value=0, max_value=len(df)-1, value=0)
    if st.button("Predict row"):
        row = df.iloc[[idx]].copy()
        st.write("Row data:")
        st.write(row.T)
        # Prepare row for regression prediction
        row_reg = row.copy()
        row_reg = row_reg.drop(['FILLING VOL', 'SKU CODE'], axis=1, errors='ignore')
        row_reg = pd.get_dummies(row_reg, drop_first=True).reindex(columns=X_reg.columns, fill_value=0)
        row_reg_scaled = scaler.transform(row_reg)
        pred_lr = m['lr'].predict(row_reg_scaled)[0]
        # polynomial
        pred_poly = m['lr_poly'].predict(m['poly'].transform(row_reg_scaled))[0]
        # classification (size) - using dt
        row_cls = row.copy()
        row_cls = row_cls.drop(['VOLUME_CATEGORY', 'SKU CODE'], axis=1, errors='ignore')
        row_cls = pd.get_dummies(row_cls, drop_first=True).reindex(columns=X_cls.columns, fill_value=0)
        pred_dt = m['dt'].predict(row_cls)[0]
        st.write(f"Linear Regression predicted FILLING VOL = {pred_lr:.4f}")
        st.write(f"Polynomial Regression predicted FILLING VOL = {pred_poly:.4f}")
        st.write(f"Decision Tree predicted VOLUME_CATEGORY = {pred_dt}")

    # Downloadable summary
    st.markdown("---")
    st.write("You can now deploy this app to Streamlit Cloud: push this repo to GitHub and select it from https://share.streamlit.io/")
    st.write("App ready. Congrats!")
else:
    st.info("Train the models using the 'Train all models' button.")
