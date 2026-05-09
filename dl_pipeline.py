"""
dl_pipeline.py
--------------
Deep-learning sentiment models built with TensorFlow / Keras:
  Model 1 – MLP on TF-IDF features
  Model 2 – 1D-CNN on learned word embeddings

Includes:
  - Text cleaning
  - TF-IDF vectorisation
  - Keras tokenisation + padding
  - Training with EarlyStopping and ReduceLROnPlateau
  - Evaluation (accuracy, F1, ROC-AUC, classification report)
  - Training-history and confusion-matrix plots
  - Demo inference on new reviews
  - Model persistence
"""

import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

import config

# Lazy-import TensorFlow so the module is importable even when TF is absent
try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv1D,
        Dense,
        Dropout,
        Embedding,
        GlobalMaxPooling1D,
        Input,
    )
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.optimizers import Adam

    # FIX: keras.preprocessing is deprecated; use tf.keras.preprocessing
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    _TF_AVAILABLE = True
    print(f"[dl_pipeline] TensorFlow {tf.__version__}")
except ImportError:
    _TF_AVAILABLE = False
    print("[dl_pipeline] WARNING: TensorFlow not found — DL models unavailable.")


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Lowercase, strip non-alpha characters, and collapse whitespace.

    Parameters
    ----------
    text : Raw review string.

    Returns
    -------
    str : Cleaned review string.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_text_data(df: pd.DataFrame):
    """
    Apply text cleaning and produce stratified train/test splits.

    Parameters
    ----------
    df : Cleaned DataFrame containing 'Review Text' and TARGET_COL.

    Returns
    -------
    X_tr, X_te  : Arrays of cleaned review strings.
    y_tr, y_te  : Label arrays.
    """
    df = df.copy()
    df["clean_review"] = df["Review Text"].apply(clean_text)

    X = df["clean_review"].values
    y = df[config.TARGET_COL].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    print(f"[dl_pipeline] Text split — Train: {len(X_tr):,} | Test: {len(X_te):,}")
    return X_tr, X_te, y_tr, y_te


# ---------------------------------------------------------------------------
# Model 1 – MLP on TF-IDF
# ---------------------------------------------------------------------------

def build_tfidf_features(X_tr, X_te):
    """
    Fit a TF-IDF vectoriser on training text and transform both splits.

    Returns
    -------
    X_tr_tfidf, X_te_tfidf : Dense numpy arrays.
    tfidf                  : Fitted TfidfVectorizer (for inference).
    """
    tfidf = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        stop_words="english",
        min_df=config.TFIDF_MIN_DF,
    )
    X_tr_tfidf = tfidf.fit_transform(X_tr).toarray()
    X_te_tfidf = tfidf.transform(X_te).toarray()
    print(f"[dl_pipeline] TF-IDF matrix shape: {X_tr_tfidf.shape}")
    return X_tr_tfidf, X_te_tfidf, tfidf


def build_mlp(input_dim: int) -> "Sequential":
    """
    Build a fully-connected MLP for binary sentiment classification.

    Architecture: Dense(512) → BN → Drop → Dense(256) → BN → Drop →
                  Dense(128) → Drop → Dense(1, sigmoid)

    Parameters
    ----------
    input_dim : Number of TF-IDF features.

    Returns
    -------
    Compiled Keras Sequential model.
    """
    model = Sequential(
        [
            Dense(512, activation="relu", input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ],
        name="MLP_Sentiment",
    )
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_mlp(X_tr_tfidf, y_tr):
    """
    Train the MLP with early stopping and LR reduction.

    Returns
    -------
    mlp_model  : Fitted Keras model.
    history    : Training History object.
    """
    mlp_model = build_mlp(X_tr_tfidf.shape[1])
    mlp_model.summary()

    callbacks = [
        EarlyStopping(patience=config.ES_PATIENCE, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=config.LR_PATIENCE, verbose=1),
    ]

    history = mlp_model.fit(
        X_tr_tfidf, y_tr,
        validation_split=0.15,
        epochs=config.DL_EPOCHS,
        batch_size=config.DL_BATCH,
        callbacks=callbacks,
        verbose=1,
    )
    return mlp_model, history


# ---------------------------------------------------------------------------
# Model 2 – 1D-CNN on word embeddings
# ---------------------------------------------------------------------------

def build_sequence_features(X_tr, X_te):
    """
    Tokenise and pad text for embedding-based models.

    Returns
    -------
    X_tr_seq, X_te_seq : Padded integer sequences.
    tokenizer          : Fitted Keras Tokenizer (for inference).
    """
    tokenizer = Tokenizer(num_words=config.VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_tr)

    X_tr_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_tr), maxlen=config.MAX_SEQ_LEN, truncating="post"
    )
    X_te_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_te), maxlen=config.MAX_SEQ_LEN, truncating="post"
    )

    print(f"[dl_pipeline] Sequence shapes — Train: {X_tr_seq.shape} | Test: {X_te_seq.shape}")
    return X_tr_seq, X_te_seq, tokenizer


def build_cnn(vocab_size: int, embed_dim: int, max_len: int) -> "Model":
    """
    Build a 1D-CNN for binary sentiment classification.

    Architecture: Embedding → Conv1D(128,5) → Conv1D(64,3) →
                  GlobalMaxPool → Dense(128) → Drop → Dense(64) →
                  Dense(1, sigmoid)

    FIX: Removed deprecated `input_length` argument from Embedding layer
         (causes warnings in Keras ≥ 2.13). Shape is inferred from the
         Input() layer instead.

    Parameters
    ----------
    vocab_size : Vocabulary size.
    embed_dim  : Embedding dimension.
    max_len    : Maximum sequence length.

    Returns
    -------
    Compiled Keras Model.
    """
    inp = Input(shape=(max_len,))
    x   = Embedding(vocab_size, embed_dim)(inp)          # input_length removed (deprecated)
    x   = Conv1D(128, 5, activation="relu")(x)
    x   = Conv1D(64,  3, activation="relu")(x)
    x   = GlobalMaxPooling1D()(x)
    x   = Dense(128, activation="relu")(x)
    x   = Dropout(0.4)(x)
    x   = Dense(64,  activation="relu")(x)
    out = Dense(1,   activation="sigmoid")(x)

    model = Model(inp, out, name="CNN_Sentiment")
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_cnn(X_tr_seq, y_tr):
    """
    Train the 1D-CNN with early stopping and LR reduction.

    Returns
    -------
    cnn_model : Fitted Keras model.
    history   : Training History object.
    """
    cnn_model = build_cnn(config.VOCAB_SIZE, config.EMBED_DIM, config.MAX_SEQ_LEN)
    cnn_model.summary()

    callbacks = [
        EarlyStopping(patience=config.ES_PATIENCE, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=config.LR_PATIENCE, verbose=1),
    ]

    history = cnn_model.fit(
        X_tr_seq, y_tr,
        validation_split=0.15,
        epochs=config.DL_EPOCHS,
        batch_size=config.DL_BATCH,
        callbacks=callbacks,
        verbose=1,
    )
    return cnn_model, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_dl_model(name: str, model, X_te, y_te) -> dict:
    """
    Evaluate a trained Keras model on test data.

    Returns
    -------
    dict with accuracy, f1, roc_auc, y_pred, y_pred_prob.
    """
    y_pred_prob = model.predict(X_te).flatten()
    y_pred      = (y_pred_prob >= 0.5).astype(int)

    acc     = accuracy_score(y_te, y_pred)
    f1      = f1_score(y_te, y_pred)
    roc_auc = roc_auc_score(y_te, y_pred_prob)

    print(f"\n── {name} Results ──")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(classification_report(y_te, y_pred, target_names=["Not Recommended", "Recommended"]))

    return {
        "name"        : name,
        "accuracy"    : acc,
        "f1"          : f1,
        "roc_auc"     : roc_auc,
        "y_pred"      : y_pred,
        "y_pred_prob" : y_pred_prob,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_dl_results(
    history_mlp,
    history_cnn,
    mlp_results: dict,
    cnn_results: dict,
    y_te,
    save_dir: str = config.OUTPUT_DIR,
) -> None:
    """
    2×3 grid of training-history curves and confusion matrices.

    Parameters
    ----------
    history_mlp, history_cnn : Keras History objects.
    mlp_results, cnn_results : Evaluation result dicts.
    y_te                     : True test labels.
    save_dir                 : Output directory.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Deep Learning Results — Women's Clothing Reviews", fontsize=14, fontweight="bold")

    # MLP accuracy
    axes[0, 0].plot(history_mlp.history["accuracy"],     label="Train Acc", color="#3498db")
    axes[0, 0].plot(history_mlp.history["val_accuracy"], label="Val Acc",   color="#e74c3c")
    axes[0, 0].set_title("MLP — Accuracy")
    axes[0, 0].legend()

    # MLP loss
    axes[0, 1].plot(history_mlp.history["loss"],     label="Train Loss", color="#3498db")
    axes[0, 1].plot(history_mlp.history["val_loss"], label="Val Loss",   color="#e74c3c")
    axes[0, 1].set_title("MLP — Loss")
    axes[0, 1].legend()

    # MLP confusion matrix
    sns.heatmap(
        confusion_matrix(y_te, mlp_results["y_pred"]),
        annot=True, fmt="d", cmap="Blues", ax=axes[0, 2],
        xticklabels=["Not Rec.", "Rec."],
        yticklabels=["Not Rec.", "Rec."],
    )
    axes[0, 2].set_title("Confusion Matrix — MLP")

    # CNN accuracy
    axes[1, 0].plot(history_cnn.history["accuracy"],     label="Train Acc", color="#2ecc71")
    axes[1, 0].plot(history_cnn.history["val_accuracy"], label="Val Acc",   color="#e67e22")
    axes[1, 0].set_title("CNN — Accuracy")
    axes[1, 0].legend()

    # CNN loss
    axes[1, 1].plot(history_cnn.history["loss"],     label="Train Loss", color="#2ecc71")
    axes[1, 1].plot(history_cnn.history["val_loss"], label="Val Loss",   color="#e67e22")
    axes[1, 1].set_title("CNN — Loss")
    axes[1, 1].legend()

    # CNN confusion matrix
    sns.heatmap(
        confusion_matrix(y_te, cnn_results["y_pred"]),
        annot=True, fmt="d", cmap="Greens", ax=axes[1, 2],
        xticklabels=["Not Rec.", "Rec."],
        yticklabels=["Not Rec.", "Rec."],
    )
    axes[1, 2].set_title("Confusion Matrix — CNN")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "dl_training_plots.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[dl_pipeline] DL plots saved → {out_path}")

    # Summary table
    print(f"\n{'Model':<22} {'Accuracy':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
    print("-" * 54)
    for r in [mlp_results, cnn_results]:
        print(
            f"{r['name']:<22} {r['accuracy']:>10.4f} "
            f"{r['f1']:>10.4f} {r['roc_auc']:>10.4f}"
        )


# ---------------------------------------------------------------------------
# Demo inference
# ---------------------------------------------------------------------------

def demo_inference(cnn_model, tokenizer) -> None:
    """
    Run the trained CNN on a few example reviews and print predictions.

    FIX vs original notebook
    ------------------------
    The original demo used a query about men's black T-shirts, which makes
    no sense for a women's clothing dataset.  Examples here are relevant.
    """
    sample_reviews = [
        "I absolutely love this dress! It fits perfectly and the material is amazing.",
        "Very disappointed. The fabric was cheap and it fell apart after one wash.",
        "Decent blouse for the price — nothing special but gets the job done.",
    ]

    sample_clean = [clean_text(r) for r in sample_reviews]
    sample_seq   = pad_sequences(
        tokenizer.texts_to_sequences(sample_clean), maxlen=config.MAX_SEQ_LEN
    )
    sample_preds = cnn_model.predict(sample_seq).flatten()

    print("\n── Demo Predictions (CNN) ──")
    for review, prob in zip(sample_reviews, sample_preds):
        label = "✔ Recommended" if prob >= 0.5 else "✘ Not Recommended"
        print(f"Review : {review}")
        print(f"Pred   : {label}  (confidence: {prob:.2%})\n")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_dl_models(mlp_model, cnn_model, save_dir: str = config.OUTPUT_DIR) -> None:
    """
    Save both Keras models.

    FIX: Use '.keras' format instead of the deprecated '.h5' extension.
         The legacy HDF5 format is deprecated in TF 2.13+ and '.keras'
         is the recommended native Keras format.
    """
    os.makedirs(save_dir, exist_ok=True)

    mlp_path = os.path.join(save_dir, "mlp_sentiment_model.keras")
    cnn_path = os.path.join(save_dir, "cnn_sentiment_model.keras")

    mlp_model.save(mlp_path)
    cnn_model.save(cnn_path)

    print(f"[dl_pipeline] MLP model saved → {mlp_path}")
    print(f"[dl_pipeline] CNN model saved → {cnn_path}")
