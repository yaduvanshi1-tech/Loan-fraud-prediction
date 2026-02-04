import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score

from xgboost import XGBClassifier

from pre_processing import load_and_clean_data
from feature_engineering import add_high_accuracy_features


DATA_PATH = "../data/defaults of credit cards and clients.csv"
MODEL_PATH = "../models/"

os.makedirs(MODEL_PATH, exist_ok=True)


def train_models():

    # ---------- Load & preprocess ----------
    df = load_and_clean_data(DATA_PATH)
    df = add_high_accuracy_features(df)

    print("Columns after preprocessing:")
    print(df.columns.tolist())

    # ---------- Target ----------
    target_col = "DEFAULT_PAYMENT_NEXT_MONTH"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ---------- Train-test split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------- Scaling (for Logistic only) ----------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------- Tree models ----------
    X_train_tree = X_train.values
    X_test_tree = X_test.values

    # ---------- Models ----------
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="auc",
        random_state=42,
        tree_method="hist"
    )

    # ---------- Train individual models ----------
    models = {
        "logistic_regression": (lr, X_train_scaled, X_test_scaled),
        "random_forest": (rf, X_train_tree, X_test_tree),
        "xgb_model": (xgb, X_train_tree, X_test_tree)
    }

    for name, (model, Xtr, Xte) in models.items():

        model.fit(Xtr, y_train)
        probs = model.predict_proba(Xte)[:, 1]
        preds = (probs > 0.5).astype(int)

        print(f"\n🔹 Model: {name}")
        print(classification_report(y_test, preds))
        print("ROC-AUC:", roc_auc_score(y_test, probs))

        joblib.dump(model, f"{MODEL_PATH}{name}.pkl")

    # ---------- ENSEMBLE (Soft Voting) ----------
    ensemble = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("rf", rf),
            ("xgb", xgb)
        ],
        voting="soft",
        weights=[1, 2, 3]  # XGB gets more weight
    )

    ensemble.fit(X_train_tree, y_train)
    ens_probs = ensemble.predict_proba(X_test_tree)[:, 1]
    ens_preds = (ens_probs > 0.5).astype(int)

    print("\n Ensemble Model (LR + RF + XGB)")
    print(classification_report(y_test, ens_preds))
    print("ROC-AUC:", roc_auc_score(y_test, ens_probs))

    joblib.dump(ensemble, f"{MODEL_PATH}ensemble.pkl")
    joblib.dump(scaler, f"{MODEL_PATH}scaler.pkl")

    print("\n Training completed. All models saved.")


if __name__ == "__main__":
    train_models()
