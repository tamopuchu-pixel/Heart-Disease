import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Heart Disease Classification App", layout="wide")

st.title("Heart Disease Classification Dashboard")
st.write("Upload the Heart Disease CSV dataset to evaluate different ML models.")


# -----------------------------
# Load Models & Scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "model")

    models = {
        "Logistic Regression": joblib.load(os.path.join(model_path, "logistic_regression.pkl")),
        "Decision Tree": joblib.load(os.path.join(model_path, "decision_tree.pkl")),
        "KNN": joblib.load(os.path.join(model_path, "knn.pkl")),
        "Naive Bayes": joblib.load(os.path.join(model_path, "naive_bayes.pkl")),
        "Random Forest": joblib.load(os.path.join(model_path, "random_forest.pkl")),
        "XGBoost": joblib.load(os.path.join(model_path, "xgboost.pkl")),
    }

    scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))

    return models, scaler


models, scaler = load_artifacts()


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("User Input")

uploaded_file = st.sidebar.file_uploader("Upload Test CSV File", type=["csv"])

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)


# -----------------------------
# Main Logic
# -----------------------------
if uploaded_file is not None:

    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("Uploaded CSV is empty.")
            st.stop()

    except Exception as e:
        st.error("Error reading CSV file. Please upload a valid CSV.")
        st.stop()

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())




    if "target" not in df.columns:
        st.error("Target column missing in uploaded file.")
        st.stop()

    X = df.drop("target", axis=1)
    y = df["target"]

    # Match training feature columns
    expected_features = scaler.feature_names_in_

    X = X.reindex(columns=expected_features, fill_value=0)

    # Scale
    X_scaled = scaler.transform(X)


    
    
    # ================================
    # Prediction
    # ================================

    model = models[selected_model_name]
    y_pred = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y, y_prob)
    else:
        auc = "Not Available"

    # ==============

        # ================================
    # Metrics Calculation
    # ================================

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    # ================================
    # Display Metrics
    # ================================

    st.subheader(f"Performance Metrics - {selected_model_name}")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")

    col1.metric("F1 Score", f"{f1:.4f}")
    col2.metric("MCC", f"{mcc:.4f}")

    if isinstance(auc, float):
        col3.metric("AUC", f"{auc:.4f}")
    else:
        col3.metric("AUC", "N/A")

        st.subheader("Classification Report")

    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)





