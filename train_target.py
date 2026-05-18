import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    classification_report
)
from sklearn.neural_network import MLPClassifier
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PCA_BASE_DIR = os.path.join(BASE_DIR, "data", "processed", "pca_variants")
RESULTS_BASE_DIR = os.path.join(BASE_DIR, "results", "target")

os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

summary_rows = []

for var in range(90, 100):

    VAR_DIR = os.path.join(PCA_BASE_DIR, f"pca_{var}")
    TRAIN_FILE = os.path.join(VAR_DIR, "target_train.xlsx")
    TEST_FILE = os.path.join(VAR_DIR, "target_test.xlsx")

    if not os.path.exists(TRAIN_FILE):
        continue

    OUT_DIR = os.path.join(RESULTS_BASE_DIR, f"pca_{var}")
    os.makedirs(OUT_DIR, exist_ok=True)

    MODEL_PATH = os.path.join(OUT_DIR, "target_model.pkl")
    ROC_PATH = os.path.join(OUT_DIR, "roc_auc.png")
    METRICS_PATH = os.path.join(OUT_DIR, "metrics.xlsx")

    train_df = pd.read_excel(TRAIN_FILE)
    test_df = pd.read_excel(TEST_FILE)

    LABEL_COL = train_df.columns[-1]

    X_train = train_df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number])
    y_train = train_df[LABEL_COL]

    X_test = test_df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number])
    y_test = test_df[LABEL_COL]

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
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

    joblib.dump(pipeline, MODEL_PATH)

    metrics_df = pd.DataFrame([{
        "pca_variance": var,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }])

    metrics_df.to_excel(METRICS_PATH, index=False)

    summary_rows.append(metrics_df.iloc[0].to_dict())

summary_df = pd.DataFrame(summary_rows)
summary_df.to_excel(
    os.path.join(RESULTS_BASE_DIR, "summary_metrics.xlsx"),
    index=False
)

print(summary_df)
