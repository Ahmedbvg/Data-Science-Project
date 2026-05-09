"""
data_loader.py
--------------
Handles:
  - Optional Kaggle dataset download (skipped if the CSV already exists locally)
  - Loading the CSV into a DataFrame
  - Exploratory data analysis (summary stats + visualisations)
"""

import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd

import config

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Download (Kaggle) — optional
# ---------------------------------------------------------------------------

def download_from_kaggle() -> None:
    """
    Download the dataset from Kaggle using the CLI.
    Requires ~/.kaggle/kaggle.json to be configured beforehand.
    Skips download if the CSV already exists locally.
    """
    csv_name = "Womens Clothing E-Commerce Reviews.csv"
    if os.path.exists(csv_name):
        print(f"[data_loader] '{csv_name}' already present — skipping download.")
        return

    print("[data_loader] Downloading dataset from Kaggle …")
    os.system("kaggle datasets download -d nicapotato/womens-ecommerce-clothing-reviews")
    os.system("unzip -o womens-ecommerce-clothing-reviews.zip")
    print("[data_loader] Download complete.")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_data(path: str = config.DATA_PATH) -> pd.DataFrame:
    """
    Read the raw CSV (first column is an unnamed index).

    Parameters
    ----------
    path : str
        File path to the CSV.

    Returns
    -------
    pd.DataFrame
        Raw, unmodified DataFrame.
    """
    df = pd.read_csv(path, index_col=0)
    print(f"[data_loader] Loaded  shape={df.shape}")
    print(f"[data_loader] Columns : {df.columns.tolist()}")
    return df


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------

def print_eda_summary(df: pd.DataFrame) -> None:
    """Print data-types, missing-value counts, and basic statistics."""
    print("\n── DATA TYPES ──────────────────────")
    print(df.dtypes)

    print("\n── MISSING VALUES ──────────────────")
    missing     = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    print(pd.DataFrame({"Count": missing, "Percent (%)": missing_pct}))

    print("\n── BASIC STATISTICS ────────────────")
    print(df.describe())

    print("\n── DISTRIBUTION SUMMARIES ──────────")
    print("Rating Distribution:")
    print(df["Rating"].value_counts().sort_index())
    print("\nRecommended Distribution:")
    print(df["Recommended IND"].value_counts())
    print("\nDepartment Distribution:")
    print(df["Department Name"].value_counts())


def plot_eda(df: pd.DataFrame, save_dir: str = config.OUTPUT_DIR) -> None:
    """
    Generate and save a 2×3 grid of EDA charts.

    Parameters
    ----------
    df       : Raw (or partially cleaned) DataFrame.
    save_dir : Directory where 'eda_plots.png' is written.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Women's E-Commerce Clothing Reviews — EDA", fontsize=16, fontweight="bold")

    # 1. Rating distribution
    rating_counts = df["Rating"].value_counts().sort_index()
    axes[0, 0].bar(
        rating_counts.index, rating_counts.values,
        color=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"],
    )
    axes[0, 0].set_title("Rating Distribution")
    axes[0, 0].set_xlabel("Rating")
    axes[0, 0].set_ylabel("Count")

    # 2. Recommended IND pie
    axes[0, 1].pie(
        df["Recommended IND"].value_counts(),
        labels=["Recommended (1)", "Not Recommended (0)"],
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
        startangle=90,
    )
    axes[0, 1].set_title("Recommended vs Not Recommended")

    # 3. Age histogram
    axes[0, 2].hist(df["Age"].dropna(), bins=30, color="#3498db", edgecolor="white")
    axes[0, 2].set_title("Age Distribution")
    axes[0, 2].set_xlabel("Age")
    axes[0, 2].set_ylabel("Count")

    # 4. Reviews per department (horizontal bar)
    dept_counts = df["Department Name"].value_counts()
    axes[1, 0].barh(dept_counts.index, dept_counts.values, color="#9b59b6")
    axes[1, 0].set_title("Reviews by Department")
    axes[1, 0].set_xlabel("Count")

    # 5. % Recommended by Rating
    rating_rec = df.groupby("Rating")["Recommended IND"].mean() * 100
    axes[1, 1].bar(
        rating_rec.index, rating_rec.values,
        color=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"],
    )
    axes[1, 1].set_title("% Recommended by Rating")
    axes[1, 1].set_xlabel("Rating")
    axes[1, 1].set_ylabel("% Recommended")

    # 6. Age by Rating (box-plot)
    axes[1, 2].boxplot(
        [df[df["Rating"] == r]["Age"].dropna() for r in range(1, 6)],
        labels=[1, 2, 3, 4, 5],
    )
    axes[1, 2].set_title("Age Distribution by Rating")
    axes[1, 2].set_xlabel("Rating")
    axes[1, 2].set_ylabel("Age")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "eda_plots.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[data_loader] EDA chart saved → {out_path}")
