"""
ml_pipeline.py
--------------
Classical machine-learning pipeline:
  1. Evaluation helper  – fits, predicts, and reports metrics for any sklearn estimator
  2. Three baseline models  – Logistic Regression, Random Forest, Gradient Boosting
  3. Hyperparameter tuning  – GridSearchCV on Random Forest
  4. Comparison visualisations  – metrics bar, ROC curves, confusion matrix
  5. Feature-importance plot
  6. Model persistence
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score

import config


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Fit a model, evaluate on the test set, and return a results dict.

    Parameters
    ----------
    name    : Human-readable model name (used in reports and plots).
    model   : Unfitted sklearn estimator.
    X_train, X_test, y_train, y_test : Data splits.

    Returns
    -------
    dict with keys: name, model, accuracy, f1, roc_auc, cv_acc,
                    y_pred, y_pred_prob
    """
    model.fit(X_train, y_train)
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    cv_acc  = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()

    print(f"\n{'─' * 55}")
    print(f"  Model          : {name}")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  F1-Score       : {f1:.4f}")
    print(f"  ROC-AUC        : {roc_auc:.4f}")
    print(f"  CV-Acc (5-fold): {cv_acc:.4f}")
    print(
        f"\n{classification_report(y_test, y_pred, target_names=['Not Recommended', 'Recommended'])}"
    )

    return {
        "name"        : name,
        "model"       : model,
        "accuracy"    : acc,
        "f1"          : f1,
        "roc_auc"     : roc_auc,
        "cv_acc"      : cv_acc,
        "y_pred"      : y_pred,
        "y_pred_prob" : y_pred_prob,
    }


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------

def run_logistic_regression(X_train, X_test, y_train, y_test) -> dict:
    """Train and evaluate a Logistic Regression model."""
    model = LogisticRegression(max_iter=config.LR_MAX_ITER, random_state=config.RANDOM_STATE)
    results = evaluate_model("Logistic Regression", model, X_train, X_test, y_train, y_test)

    # Show top features by absolute coefficient
    feat_df = pd.DataFrame({
        "Feature"    : X_train.columns,
        "Coefficient": np.abs(model.coef_[0]),
    }).sort_values("Coefficient", ascending=False)
    print("\nTop features (abs coefficient):")
    print(feat_df.to_string(index=False))

    return results


def run_random_forest(X_train, X_test, y_train, y_test) -> dict:
    """Train and evaluate a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    results = evaluate_model("Random Forest", model, X_train, X_test, y_train, y_test)

    feat_df = pd.DataFrame({
        "Feature"   : X_train.columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    print("\nFeature Importance (Random Forest):")
    print(feat_df.to_string(index=False))

    return results


def run_gradient_boosting(X_train, X_test, y_train, y_test) -> dict:
    """Train and evaluate a Gradient Boosting classifier."""
    model = GradientBoostingClassifier(
        n_estimators=config.GB_N_ESTIMATORS,
        learning_rate=config.GB_LEARNING_RATE,
        max_depth=config.GB_MAX_DEPTH,
        random_state=config.RANDOM_STATE,
    )
    return evaluate_model("Gradient Boosting", model, X_train, X_test, y_train, y_test)


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_random_forest(X_train, X_test, y_train, y_test) -> dict:
    """
    Run GridSearchCV on Random Forest and evaluate the best estimator.

    Uses ROC-AUC as the optimisation metric.
    """
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
        config.RF_PARAM_GRID,
        cv=3,
        scoring="roc_auc",
        verbose=1,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    print(f"[ml_pipeline] Best Parameters: {grid_search.best_params_}")
    print(f"[ml_pipeline] Best ROC-AUC   : {grid_search.best_score_:.4f}")

    best_rf = grid_search.best_estimator_
    return evaluate_model("Random Forest (Tuned)", best_rf, X_train, X_test, y_train, y_test)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_model_comparison(all_results: list, y_test: pd.Series, save_dir: str = config.OUTPUT_DIR) -> None:
    """
    Produce and save a 3-panel comparison figure:
      Panel 1 – grouped bar chart of Accuracy / F1 / ROC-AUC
      Panel 2 – ROC curves for all models
      Panel 3 – Confusion matrix for the best model by ROC-AUC

    Parameters
    ----------
    all_results : List of result dicts from evaluate_model().
    y_test      : True test labels.
    save_dir    : Output directory.
    """
    os.makedirs(save_dir, exist_ok=True)

    comparison_df = pd.DataFrame([
        {
            "Model"   : r["name"],
            "Accuracy": round(r["accuracy"], 4),
            "F1-Score": round(r["f1"],       4),
            "ROC-AUC" : round(r["roc_auc"],  4),
            "CV-Acc"  : round(r["cv_acc"],   4),
        }
        for r in all_results
    ])

    print("\n=== MODEL COMPARISON ===")
    print(comparison_df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("ML Model Comparison — Women's Clothing Reviews", fontsize=14, fontweight="bold")

    # Panel 1 – metrics bar chart
    metrics = ["Accuracy", "F1-Score", "ROC-AUC"]
    x       = np.arange(len(comparison_df))
    width   = 0.25
    colors  = ["#3498db", "#2ecc71", "#e74c3c"]
    for i, metric in enumerate(metrics):
        axes[0].bar(x + i * width, comparison_df[metric], width, label=metric, color=colors[i], alpha=0.85)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(comparison_df["Model"], rotation=20, ha="right", fontsize=9)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title("Metrics Comparison")
    axes[0].legend()

    # Panel 2 – ROC curves
    for r in all_results:
        fpr, tpr, _ = roc_curve(y_test, r["y_pred_prob"])
        axes[1].plot(fpr, tpr, label=f'{r["name"]} (AUC={r["roc_auc"]:.3f})', linewidth=2)
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curves")
    axes[1].legend(fontsize=8)

    # Panel 3 – Confusion matrix for best model
    best = max(all_results, key=lambda r: r["roc_auc"])
    cm   = confusion_matrix(y_test, best["y_pred"])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
        xticklabels=["Not Rec.", "Rec."],
        yticklabels=["Not Rec.", "Rec."],
    )
    axes[2].set_title(f"Confusion Matrix — {best['name']}")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "ml_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[ml_pipeline] Comparison chart saved → {out_path}")


def plot_feature_importance(model, feature_names, save_dir: str = config.OUTPUT_DIR) -> None:
    """
    Plot and save a horizontal bar chart of feature importances.

    Parameters
    ----------
    model         : Fitted tree-based model with .feature_importances_.
    feature_names : Column names corresponding to importances.
    save_dir      : Output directory.
    """
    os.makedirs(save_dir, exist_ok=True)

    feat_df = pd.DataFrame({
        "Feature"   : feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    plt.figure(figsize=(10, 5))
    plt.barh(feat_df["Feature"], feat_df["Importance"], color="#9b59b6")
    plt.title("Feature Importance — Tuned Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()

    out_path = os.path.join(save_dir, "feature_importance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[ml_pipeline] Feature importance chart saved → {out_path}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_best_model(model, save_dir: str = config.OUTPUT_DIR) -> str:
    """
    Serialise the best ML model with joblib.

    Returns
    -------
    str : Path where the model was saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "best_ml_model.pkl")
    joblib.dump(model, path)
    print(f"[ml_pipeline] Best model saved → {path}")
    return path


def load_ml_splits(save_dir: str = config.OUTPUT_DIR):
    """
    Reload train/test splits that were written by preprocessing.save_splits().

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train = pd.read_csv(os.path.join(save_dir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(save_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(save_dir, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(save_dir, "y_test.csv")).squeeze()
    print(f"[ml_pipeline] Loaded splits — X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test
