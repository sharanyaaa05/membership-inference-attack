import os
import random
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_BASE_DIR = "data/processed/pca_variants"
SAVE_BASE_DIR = "models/shadow_models"
ATTACK_BASE_DIR = "data/processed/pca_variants"

NUM_SHADOW_MODELS = 20
TRAIN_RATIO = 0.7
RANDOM_STATE = 42
EPS = 1e-12

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def entropy(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, EPS, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)

for var in range(90, 100):

    VAR_DIR = os.path.join(DATA_BASE_DIR, f"pca_{var}")
    SHADOW_POOL_FILE = os.path.join(VAR_DIR, "shadow_train.xlsx")

    if not os.path.exists(SHADOW_POOL_FILE):
        continue

    SAVE_DIR = os.path.join(SAVE_BASE_DIR, f"pca_{var}")
    ATTACK_DIR = os.path.join(ATTACK_BASE_DIR, f"pca_{var}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(ATTACK_DIR, exist_ok=True)

    df = pd.read_excel(SHADOW_POOL_FILE).reset_index(drop=True)
    LABEL_COL = df.columns[-1]

    X = df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number])
    y = df[LABEL_COL].astype(int).values

    attack_rows = []

    for shadow_id in tqdm(range(NUM_SHADOW_MODELS), desc=f"Training shadow models PCA {var}"):

        seed_everything(RANDOM_STATE + shadow_id)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=TRAIN_RATIO,
            stratify=y,
            random_state=RANDOM_STATE + shadow_id
        )

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        pca = PCA(n_components=var / 100.0, svd_solver="full", random_state=42)

        X_train = imputer.fit_transform(X_train)
        X_train = scaler.fit_transform(X_train)
        X_train = pca.fit_transform(X_train)

        X_test = imputer.transform(X_test)
        X_test = scaler.transform(X_test)
        X_test = pca.transform(X_test)

        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            random_state=RANDOM_STATE + shadow_id
        )

        model.fit(X_train, y_train)

        probs_mem = model.predict_proba(X_train)
        preds_mem = model.predict(X_train)

        probs_non = model.predict_proba(X_test)
        preds_non = model.predict(X_test)

        k = min(len(X_train), len(X_test))
        mem_idx = np.random.choice(len(X_train), k, replace=False)
        non_idx = np.random.choice(len(X_test), k, replace=False)

        for i in mem_idx:
            attack_rows.append({
                "max_prob": probs_mem[i].max(),
                "entropy": entropy(probs_mem[i:i+1])[0],
                "loss": -np.log(probs_mem[i, y_train[i]] + EPS),
                "correct": int(preds_mem[i] == y_train[i]),
                "is_member": 1
            })

        for i in non_idx:
            attack_rows.append({
                "max_prob": probs_non[i].max(),
                "entropy": entropy(probs_non[i:i+1])[0],
                "loss": -np.log(probs_non[i, y_test[i]] + EPS),
                "correct": int(preds_non[i] == y_test[i]),
                "is_member": 0
            })

        joblib.dump(
            {
                "model": model,
                "imputer": imputer,
                "scaler": scaler,
                "pca": pca
            },
            os.path.join(SAVE_DIR, f"shadow_mlp_{shadow_id}.pkl")
        )

    attack_df = pd.DataFrame(attack_rows)
    attack_df = attack_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    attack_path = os.path.join(ATTACK_DIR, "attack_dataset.xlsx")
    attack_df.to_excel(attack_path, index=False)

    print(f"\nAttack dataset saved for PCA {var}%:", attack_path)
    print("Final attack dataset shape:", attack_df.shape)
    print("Member distribution:\n", attack_df["is_member"].value_counts())
