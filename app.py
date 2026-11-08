"""
Streamlit app: Breast Cancer classification with 5 ML models.

Run locally:  streamlit run app.py
"""
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TARGET = "target"
CLASS_NAMES = ["Malignant (0)", "Benign (1)"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl",
}


@st.cache_resource
def load_model(fname):
    return joblib.load(os.path.join(MODEL_DIR, fname))


@st.cache_resource
def load_feature_names():
    return joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))


def compute_metrics(y_true, y_pred, y_prob):
    m = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    # AUC needs both classes present in y_true
    m["AUC"] = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    return m


st.set_page_config(page_title="ML Assignment 2 - Classifier", layout="wide")
st.title("Breast Cancer Classification - Model Comparison")
st.write(
    "Dataset: **Breast Cancer Wisconsin (Diagnostic)**, UCI ML Repository - "
    "30 numeric features, 569 instances, binary target (0 = malignant, 1 = benign)."
)

# ---- Sidebar: model selection + upload ----
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
uploaded = st.sidebar.file_uploader(
    "Upload test data (CSV)", type=["csv"],
    help="Use test_data.csv from the repo, or any CSV with the same 30 feature "
         "columns. Include a 'target' column to see evaluation metrics.")

# ---- Benchmark table from training run ----
metrics_path = os.path.join(MODEL_DIR, "metrics.csv")
if os.path.exists(metrics_path):
    st.subheader("Benchmark metrics on held-out test set (all models)")
    st.dataframe(pd.read_csv(metrics_path, index_col=0).style.format("{:.4f}"),
                 use_container_width=True)

if uploaded is None:
    st.info("Upload a CSV in the sidebar (e.g. test_data.csv) to run predictions.")
    st.stop()

df = pd.read_csv(uploaded)
features = load_feature_names()
missing = [c for c in features if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing {len(missing)} required column(s): {missing[:5]}...")
    st.stop()

X = df[features]
model = load_model(MODEL_FILES[model_name])
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

st.subheader(f"Results - {model_name}")

if TARGET in df.columns:
    y_true = df[TARGET].astype(int)
    m = compute_metrics(y_true, y_pred, y_prob)
    cols = st.columns(6)
    for col, key in zip(cols, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
        col.metric(key, f"{m[key]:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("**Classification Report**")
        report = classification_report(y_true, y_pred, labels=[0, 1],
                                       target_names=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)
else:
    st.warning("No 'target' column found - showing predictions only.")

out = df.copy()
out["prediction"] = y_pred
out["prob_benign"] = y_prob.round(4)
st.markdown("**Predictions**")
st.dataframe(out, use_container_width=True)
st.download_button("Download predictions as CSV", out.to_csv(index=False),
                   file_name="predictions.csv", mime="text/csv")
