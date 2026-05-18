import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    classification_report
)
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PCA_BASE_DIR = os.path.join(BASE_DIR, "data", "processed", "pca_variants")
RESULTS_BASE_DIR = os.path.join(BASE_DIR, "results", "attack")

os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

summary_rows = []

for var in range(90, 100):

    VAR_DIR = os.path.join(PCA_BASE_DIR, f"pca_{var}")
    ATTACK_DATA_PATH = os.path.join(VAR_DIR, "attack_dataset.xlsx")

    if not os.path.exists(ATTACK_DATA_PATH):
        continue

    OUT_DIR = os.path.join(RESULTS_BASE_DIR, f"pca_{var}")
    os.makedirs(OUT_DIR, exist_ok=True)

    ATTACK_MODEL_PATH = os.path.join(OUT_DIR, "attack_model.pkl")
    ROC_PATH = os.path.join(OUT_DIR, "attack_roc_auc.png")
    METRICS_PATH = os.path.join(OUT_DIR, "attack_metrics.xlsx")

    df = pd.read_excel(ATTACK_DATA_PATH)

    y = df["is_member"].astype(int)
    X = df.drop(columns=["is_member"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig(ROC_PATH)
    plt.close()

    metrics_df = pd.DataFrame([{
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }])

    metrics_df.to_excel(METRICS_PATH, index=False)
    joblib.dump(pipeline, ATTACK_MODEL_PATH)

    summary_rows.append({
        "pca_variance": var,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_excel(
    os.path.join(RESULTS_BASE_DIR, "summary_auc.xlsx"),
    index=False
)
