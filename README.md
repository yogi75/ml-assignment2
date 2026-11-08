# ML Assignment 2 — Breast Cancer Classification

## a. Problem Statement
Predict whether a breast tumour is **malignant** or **benign** from features computed from a digitized image of a fine-needle aspirate (FNA) of a breast mass. Five classification models are trained on the same dataset, evaluated with six metrics, and served through an interactive Streamlit app.

## b. Dataset Description
- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (loaded with `sklearn.datasets.load_breast_cancer`, which ships an identical copy)
- **Type:** Binary classification
- **Instances:** 569 (357 benign, 212 malignant)
- **Features:** 30 real-valued features (requirement: ≥ 12). For each of 10 cell-nucleus properties (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) the dataset gives the **mean**, **standard error** and **worst** (largest) value.
- **Target:** `target` — 0 = malignant, 1 = benign
- **Missing values:** none

**Preprocessing:** stratified 80/20 train/test split (`random_state=42`), then `StandardScaler` fitted on the training data. The scaler is saved inside each model pipeline, so every `.pkl` file accepts raw feature values.

## c. Models Used & Evaluation Metrics
All metrics are computed on the 114-row held-out test set (`test_data.csv`). The positive class is 1 (benign).

| ML Model Name       | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|---------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree       | 0.9211   | 0.9163 | 0.9565    | 0.9167 | 0.9362 | 0.8341 |
| kNN                 | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes         | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| SVM                 | 0.9825   | 0.9950 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |

### Observations

| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Best overall (tied with SVM): 98.25% accuracy and the highest AUC (0.9954). The classes are nearly linearly separable once features are scaled, so a simple linear model generalises very well with little overfitting. |
| Decision Tree       | Weakest model (92.1% accuracy, MCC 0.834, lowest AUC 0.916). Even with depth limited to 5, a single tree makes axis-aligned splits that fit the training data closely and generalise less well. Its predicted probabilities are coarse (leaf proportions), which lowers its AUC. |
| kNN                 | Strong (95.6% accuracy, MCC 0.905) because scaling makes distances meaningful. Recall is high (0.972), but it does worse than the linear models, likely because distance-based voting struggles in 30 dimensions. |
| Naive Bayes         | Fast and decent (93.0% accuracy), with a high AUC (0.987), so its ranking of cases is good. The independence assumption is violated (radius, perimeter and area are strongly correlated), which hurts its hard predictions and MCC (0.849). |
| SVM                 | Tied for best (98.25% accuracy, MCC 0.962, AUC 0.995). The RBF kernel with maximum-margin training handles the scaled, high-dimensional features very well and is robust to outliers. |

**Summary:** Logistic Regression and SVM are the most reliable models for this dataset. MCC separates the models more clearly than accuracy does, because the classes are mildly imbalanced (63% / 37%).

## Repository Structure
```
ml-assignment-2/
│-- app.py                    # Streamlit app
│-- requirements.txt
│-- README.md
│-- test_data.csv             # held-out test split (30 features + target)
│-- model/
    │-- train_models.py       # training + evaluation source code
    │-- logistic_regression.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    │-- svm.pkl
    │-- feature_names.pkl     # feature column order (preprocessing artifact)
    │-- metrics.csv           # metrics table above
```

## How to Run
```bash
pip install -r requirements.txt
python model/train_models.py   # optional: retrains models, regenerates metrics + test_data.csv
streamlit run app.py
```

## Streamlit App Features
- CSV upload (use `test_data.csv`)
- Drop-down to pick any of the 5 models
- Display of all 6 evaluation metrics, plus a benchmark table for every model
- Confusion matrix and classification report
- Downloadable predictions

## Deployment (Streamlit Community Cloud)
1. Push this folder to a public GitHub repository.
2. Go to https://streamlit.io/cloud, choose **New app**, and select the repo, branch and `app.py`.
3. Deploy. `scikit-learn` is pinned to 1.9.1 so the saved `.pkl` models load correctly.
