import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve
)

from pre_processing import load_and_clean_data
from feature_engineering import add_high_accuracy_features


DATA_PATH = "../data/defaults of credit cards and clients.csv"
MODEL_PATH = "../models/"
TARGET_COL = "DEFAULT_PAYMENT_NEXT_MONTH"


def evaluate_model(model_name, threshold=0.5, use_scaler=False):

    # ---------- Load data ----------
    df = load_and_clean_data(DATA_PATH)
    df = add_high_accuracy_features(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # ---------- Load model ----------
    model = joblib.load(f"{MODEL_PATH}{model_name}.pkl")

    # ---------- Scaling if needed ----------
    if use_scaler:
        scaler = joblib.load(f"{MODEL_PATH}scaler.pkl")
        X = scaler.transform(X)
    else:
        X = X.values

    # ---------- Predictions ----------
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    # ---------- Metrics ----------
    print(f"\n Evaluating Model: {model_name}")
    print(f"Threshold: {threshold}")

    print("\n Classification Report:")
    print(classification_report(y, preds))

    print(" Confusion Matrix:")
    print(confusion_matrix(y, preds))

    print("ROC-AUC:", roc_auc_score(y, probs))


def find_best_threshold(model_name, use_scaler=False):
    """
    Finds threshold maximizing F1-score
    """

    df = load_and_clean_data(DATA_PATH)
    df = add_high_accuracy_features(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    model = joblib.load(f"{MODEL_PATH}{model_name}.pkl")

    if use_scaler:
        scaler = joblib.load(f"{MODEL_PATH}scaler.pkl")
        X = scaler.transform(X)
    else:
        X = X.values

    probs = model.predict_proba(X)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"\n Best Threshold for {model_name}: {best_threshold:.3f}")
    print(f"Precision: {precision[best_idx]:.3f}")
    print(f"Recall: {recall[best_idx]:.3f}")
    print(f"F1-score: {f1_scores[best_idx]:.3f}")

    return best_threshold


if __name__ == "__main__":
    
    evaluate_model("logistic_regression", threshold=0.5, use_scaler=True)
    evaluate_model("random_forest", threshold=0.5)
    evaluate_model("xgb_model", threshold=0.4)


    # ----- Threshold tuning (recommended) -----
    find_best_threshold("xgb_model")
    find_best_threshold("ensemble")
