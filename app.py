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
st.set_page_config(
    page_title="ML Classification App",
    layout="wide"
)

st.title("Machine Learning Classification Dashboard")
st.write("Upload test dataset and evaluate selected model.")

# -----------------------------
# Load Models and Scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl")
    }

    scaler = joblib.load("model/scaler.pkl")

    return models, scaler

models, scaler = load_artifacts()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("User Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

# -----------------------------
# Main Logic
# -----------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Remove any unwanted index column if exists
    if "Unnamed: 0" in X.columns:
        X = X.drop(columns=["Unnamed: 0"])

    # Apply scaling
    X_scaled = scaler.transform(X.values)

    model = models[selected_model_name]

    # Prediction
    y_pred = model.predict(X_scaled)

    # AUC calculation
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y, y_prob)
    else:
        auc = "Not Available"

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    mcc = matthews_corrcoef(y, y_pred)

    # -----------------------------
    # Display Metrics
    # -----------------------------
    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{f1:.4f}")
    col5.metric("MCC", f"{mcc:.4f}")
    col6.metric("AUC", auc if isinstance(auc, str) else f"{auc:.4f}")

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # -----------------------------
    # Classification Report
    # -----------------------------
    st.subheader("Classification Report")

    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

else:
    st.info("Please upload a test CSV file to begin.")


