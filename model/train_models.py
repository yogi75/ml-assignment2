"""
Train and evaluate 5 classification models on the Breast Cancer Wisconsin
(Diagnostic) dataset (UCI ML Repository).

Run from the project root:
    python model/train_models.py

Outputs (written to model/):
    - logistic_regression.pkl, decision_tree.pkl, knn.pkl,
      naive_bayes.pkl, svm.pkl          -> trained pipelines (scaler + model)
    - feature_names.pkl                 -> feature column order used in training
    - metrics.csv                       -> evaluation metrics for all models
and test_data.csv in the project root (held-out test split with labels).
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODEL_DIR)
TARGET = "target"

MODELS = {
    "Logistic Regression": ("logistic_regression.pkl",
                            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    "Decision Tree": ("decision_tree.pkl",
                      DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
    "KNN": ("knn.pkl", KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes": ("naive_bayes.pkl", GaussianNB()),
    "SVM": ("svm.pkl", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
}


def load_data():
    """Breast Cancer Wisconsin (Diagnostic): 569 rows, 30 numeric features.
    Target: 1 = benign, 0 = malignant."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    return df.drop(columns=[TARGET]), df[TARGET]


def evaluate(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    X, y = load_data()
    print(f"Dataset shape: {X.shape}, class counts: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Save held-out test split (features + label) for use in the Streamlit app
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(os.path.join(PROJECT_DIR, "test_data.csv"), index=False)
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.pkl"))

    rows = []
    for name, (fname, clf) in MODELS.items():
        # StandardScaler is the preprocessing step; bundled in the pipeline so the
        # saved .pkl can be applied directly to raw feature values.
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_pred, y_prob)
        rows.append({"Model": name, **metrics})
        joblib.dump(pipe, os.path.join(MODEL_DIR, fname))
        print(f"Saved {fname}")

    results = pd.DataFrame(rows).set_index("Model").round(4)
    results.to_csv(os.path.join(MODEL_DIR, "metrics.csv"))
    print("\n", results.to_string())


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
