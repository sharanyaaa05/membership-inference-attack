import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

INPUT_DATA_PATH = "data/processed/features_df.csv"
OUTPUT_BASE_DIR = "data/processed/pca_variants"

RANDOM_STATE = 42
TARGET_RATIO = 0.5
TARGET_TEST_SIZE = 0.3
SHADOW_TEST_SIZE = 0.3
LABEL_COL = "label"

PCA_VARIANCES = list(range(90, 100))

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

df = pd.read_csv(INPUT_DATA_PATH)
df = shuffle(df, random_state=RANDOM_STATE).reset_index(drop=True)

assert df[LABEL_COL].nunique() == 2

df_target, df_shadow = train_test_split(
    df,
    test_size=1 - TARGET_RATIO,
    stratify=df[LABEL_COL],
    random_state=RANDOM_STATE
)

target_train, target_test = train_test_split(
    df_target,
    test_size=TARGET_TEST_SIZE,
    stratify=df_target[LABEL_COL],
    random_state=RANDOM_STATE
)

shadow_train, shadow_test = train_test_split(
    df_shadow,
    test_size=SHADOW_TEST_SIZE,
    stratify=df_shadow[LABEL_COL],
    random_state=RANDOM_STATE
)

for var in PCA_VARIANCES:
    print(f"\nProcessing PCA {var}% ")

    out_dir = os.path.join(OUTPUT_BASE_DIR, f"pca_{var}")
    os.makedirs(out_dir, exist_ok=True)

    X_target_train = target_train.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    y_target_train = target_train[LABEL_COL]

    X_target_test = target_test.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    y_target_test = target_test[LABEL_COL]

    X_shadow_train = shadow_train.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    y_shadow_train = shadow_train[LABEL_COL]

    X_shadow_test = shadow_test.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    y_shadow_test = shadow_test[LABEL_COL]

    imputer = SimpleImputer(strategy="median")
    X_target_train = imputer.fit_transform(X_target_train)
    X_target_test = imputer.transform(X_target_test)
    X_shadow_train = imputer.transform(X_shadow_train)
    X_shadow_test = imputer.transform(X_shadow_test)

    scaler = StandardScaler()
    X_target_train_scaled = scaler.fit_transform(X_target_train)
    X_target_test_scaled = scaler.transform(X_target_test)
    X_shadow_train_scaled = scaler.transform(X_shadow_train)
    X_shadow_test_scaled = scaler.transform(X_shadow_test)

    pca = PCA(n_components=var / 100.0, random_state=RANDOM_STATE)
    X_target_train_pca = pca.fit_transform(X_target_train_scaled)
    X_target_test_pca = pca.transform(X_target_test_scaled)
    X_shadow_train_pca = pca.transform(X_shadow_train_scaled)
    X_shadow_test_pca = pca.transform(X_shadow_test_scaled)

    def rebuild_df(X_pca, y):
        cols = [f"pc_{i}" for i in range(X_pca.shape[1])]
        df_pca = pd.DataFrame(X_pca, columns=cols)
        df_pca[LABEL_COL] = y.values
        return df_pca

    rebuild_df(X_target_train_pca, y_target_train).to_excel(os.path.join(out_dir, "target_train.xlsx"), index=False)
    rebuild_df(X_target_test_pca, y_target_test).to_excel(os.path.join(out_dir, "target_test.xlsx"), index=False)
    rebuild_df(X_shadow_train_pca, y_shadow_train).to_excel(os.path.join(out_dir, "shadow_train.xlsx"), index=False)
    rebuild_df(X_shadow_test_pca, y_shadow_test).to_excel(os.path.join(out_dir, "shadow_test.xlsx"), index=False)

    print(f"Saved PCA {var}% datasets → {out_dir}")

print("\nAll PCA datasets generated successfully.")
