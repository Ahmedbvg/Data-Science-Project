"""
preprocessing.py
----------------
Handles all data-preparation steps:
  1. Data cleaning     – drop/fill missing values
  2. Feature engineering – text lengths, age groups
  3. Encoding & scaling  – LabelEncoder per category column, StandardScaler
  4. Train / test split
  5. Class-imbalance fix – SMOTE oversampling + class-weight computation
  6. Persisting processed splits to CSV
"""

import os
from collections import Counter
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import compute_class_weight

import config


# ---------------------------------------------------------------------------
# 1. Data cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unusable rows and fill remaining missing values.

    Steps
    -----
    - Drop rows where 'Review Text' is missing (required for NLP tasks).
    - Fill missing 'Title' with empty string.
    - Fill categorical columns with column mode.
    - Fill missing 'Age' with column median.

    Parameters
    ----------
    df : Raw DataFrame from data_loader.

    Returns
    -------
    pd.DataFrame
        Cleaned copy — original is never modified.
    """
    df_clean = df.copy()

    # Drop rows with no review text
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=["Review Text"])
    print(f"[preprocessing] Removed {before - len(df_clean)} rows with missing 'Review Text'")

    # Fill empty title
    df_clean["Title"] = df_clean["Title"].fillna("")

    # Fill categorical columns with mode
    for col in config.CAT_COLS:
        mode_val = df_clean[col].mode()[0]
        df_clean[col] = df_clean[col].fillna(mode_val)
        print(f"[preprocessing] Filled '{col}' → mode: '{mode_val}'")

    # Fill Age with median
    median_age = df_clean["Age"].median()
    df_clean["Age"] = df_clean["Age"].fillna(median_age)
    print(f"[preprocessing] Filled 'Age' → median: {median_age}")

    remaining = df_clean.isnull().sum().sum()
    print(f"[preprocessing] Remaining NaN cells: {remaining}")
    print(f"[preprocessing] Clean dataset shape : {df_clean.shape}")
    return df_clean


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive new features from existing columns.

    New columns added
    -----------------
    Review_Length     : character count of review text
    Review_Word_Count : word count of review text
    Title_Length      : character count of title
    Age_Group         : binned age category

    Parameters
    ----------
    df : Cleaned DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with new feature columns appended.
    """
    df = df.copy()

    df["Review_Length"]     = df["Review Text"].apply(len)
    df["Review_Word_Count"] = df["Review Text"].apply(lambda x: len(x.split()))
    df["Title_Length"]      = df["Title"].apply(len)

    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"],
    )

    print("[preprocessing] Added features: Review_Length, Review_Word_Count, Title_Length, Age_Group")
    return df


# ---------------------------------------------------------------------------
# 3. Encoding & scaling
# ---------------------------------------------------------------------------

def encode_and_scale(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, StandardScaler, Dict[str, LabelEncoder]]:
    """
    Label-encode categorical columns and standard-scale the feature matrix.

    FIX vs original notebook
    ------------------------
    The original used a single LabelEncoder object for all columns in a loop,
    so it only retained the mapping for the *last* column — making it impossible
    to inverse-transform or reuse encoders for inference.
    Here we keep one encoder per column in a dict.

    Parameters
    ----------
    df : DataFrame after feature engineering.

    Returns
    -------
    X_scaled   : Scaled feature matrix (DataFrame).
    y          : Target series.
    scaler     : Fitted StandardScaler (for inference).
    encoders   : Dict {column_name: fitted LabelEncoder} (for inference).
    """
    df_ml    = df.copy()
    encoders: Dict[str, LabelEncoder] = {}

    # Label-encode each categorical column separately
    for col in config.CAT_COLS:
        le = LabelEncoder()
        df_ml[f"{col}_enc"] = le.fit_transform(df_ml[col].astype(str))
        encoders[col] = le
        print(f"[preprocessing] Encoded: '{col}' → '{col}_enc'  ({len(le.classes_)} classes)")

    X = df_ml[config.ML_FEATURES]
    y = df_ml[config.TARGET_COL]

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=config.ML_FEATURES)

    print(f"[preprocessing] Feature matrix : {X_scaled.shape}")
    print(f"[preprocessing] Target balance : {y.value_counts(normalize=True).round(3).to_dict()}")
    return X_scaled, y, scaler, encoders


# ---------------------------------------------------------------------------
# 4. Train / test split
# ---------------------------------------------------------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified 80/20 train-test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    print(f"[preprocessing] Train: {X_train.shape[0]:,} samples | Test: {X_test.shape[0]:,} samples")
    print(f"[preprocessing] Train balance: {y_train.value_counts(normalize=True).round(3).to_dict()}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 5. Class-imbalance handling
# ---------------------------------------------------------------------------

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Oversample the minority class with SMOTE.

    Prints before/after class distributions.

    Parameters
    ----------
    X_train, y_train : Training split.

    Returns
    -------
    X_resampled, y_resampled : Balanced training set.
    """
    print("\n[preprocessing] Original class distribution:")
    for cls, cnt in sorted(Counter(y_train).items()):
        bar = "█" * (cnt // 100)
        pct = cnt / len(y_train) * 100
        print(f"  Class {cls}: {cnt:>5} samples  ({pct:.1f}%)  {bar}")

    smote = SMOTE(random_state=config.RANDOM_STATE, k_neighbors=config.SMOTE_K_NEIGHBORS)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print("\n[preprocessing] After SMOTE:")
    for cls, cnt in sorted(Counter(y_res).items()):
        bar = "█" * (cnt // 100)
        pct = cnt / len(y_res) * 100
        print(f"  Class {cls}: {cnt:>5} samples  ({pct:.1f}%)  {bar}")

    print(f"  Size {X_train.shape[0]:,} → {X_res.shape[0]:,} samples")
    return X_res, y_res


def compute_class_weights(y_train: pd.Series) -> Dict[int, float]:
    """
    Compute balanced class weights for models that accept a class_weight argument.

    Returns
    -------
    Dict mapping class label to weight float.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = dict(zip(classes, weights))
    print("[preprocessing] Class weights:", {k: round(v, 4) for k, v in cw_dict.items()})
    return cw_dict


# ---------------------------------------------------------------------------
# 6. Persist processed data
# ---------------------------------------------------------------------------

def save_splits(
    df_clean: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    save_dir: str = config.OUTPUT_DIR,
) -> None:
    """
    Write processed splits to CSV files.

    Parameters
    ----------
    df_clean            : Cleaned, feature-engineered DataFrame.
    X_train, X_test     : Feature splits.
    y_train, y_test     : Target splits.
    save_dir            : Directory to write CSVs into.
    """
    os.makedirs(save_dir, exist_ok=True)

    df_clean.to_csv(os.path.join(save_dir, "processed_reviews.csv"), index=False)
    X_train.to_csv(os.path.join(save_dir, "X_train.csv"),            index=False)
    X_test.to_csv(os.path.join(save_dir, "X_test.csv"),              index=False)
    y_train.to_csv(os.path.join(save_dir, "y_train.csv"),            index=False)
    y_test.to_csv(os.path.join(save_dir, "y_test.csv"),              index=False)

    print(f"[preprocessing] Splits saved to '{save_dir}/'")
