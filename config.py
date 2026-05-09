"""
config.py
---------
Central configuration for the Women's E-Commerce Clothing Reviews pipeline.
All tuneable constants live here so no magic numbers are scattered across modules.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = os.getenv(
    "DATA_PATH",
    "Womens Clothing E-Commerce Reviews.csv",   # override via env var when needed
)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")   # all saved artefacts go here

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.20   # 80 / 20 train-test split
TARGET_COL   = "Recommended IND"

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------
TEXT_COLS = ["Title", "Review Text"]
CAT_COLS  = ["Division Name", "Department Name", "Class Name"]

ML_FEATURES = [
    "Age", "Rating", "Positive Feedback Count",
    "Review_Length", "Review_Word_Count", "Title_Length",
    "Division Name_enc", "Department Name_enc", "Class Name_enc",
]

# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------
SMOTE_K_NEIGHBORS = 5

# ---------------------------------------------------------------------------
# Machine-learning hyper-parameters
# ---------------------------------------------------------------------------
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH     = 10
GB_N_ESTIMATORS  = 200
GB_LEARNING_RATE = 0.1
GB_MAX_DEPTH     = 5
LR_MAX_ITER      = 1000

# GridSearchCV param grid for Random Forest
RF_PARAM_GRID = {
    "n_estimators"     : [100, 200],
    "max_depth"        : [8, 12, None],
    "min_samples_split": [2, 5],
}

# ---------------------------------------------------------------------------
# Deep-learning hyper-parameters
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE  = (1, 2)
TFIDF_MIN_DF       = 3

VOCAB_SIZE  = 20_000
MAX_SEQ_LEN = 150
EMBED_DIM   = 64
DL_EPOCHS   = 30
DL_BATCH    = 256
ES_PATIENCE = 5
LR_PATIENCE = 3

# ---------------------------------------------------------------------------
# RAG / LLM
# ---------------------------------------------------------------------------
# FIX: API key must come from environment — never hard-code credentials.
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL     = "gemini-2.5-flash"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K            = 8
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "reviews.index")
METADATA_PATH    = os.path.join(OUTPUT_DIR, "metadata.json")
