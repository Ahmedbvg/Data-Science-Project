# Women's E-Commerce Clothing Reviews — ML / DL / RAG Pipeline

End-to-end sentiment and recommendation system built on the
[Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
dataset from Kaggle.

---

## Project structure

```
clothing_review_project/
│
├── config.py           # All tuneable constants (paths, hyper-params, API settings)
├── data_loader.py      # Dataset download (Kaggle), loading, EDA + plots
├── preprocessing.py    # Cleaning, feature engineering, encoding, SMOTE, splits
├── ml_pipeline.py      # Logistic Regression, Random Forest, Gradient Boosting + GridSearch
├── dl_pipeline.py      # MLP (TF-IDF) + 1D-CNN (word embeddings) with Keras
├── rag_pipeline.py     # FAISS vector index + Gemini LLM RAG query system
├── main.py             # Pipeline orchestrator (CLI entry point)
│
├── requirements.txt
└── README.md
```

All generated files (plots, models, processed CSVs, FAISS index) are written
to the `outputs/` directory (configurable via the `OUTPUT_DIR` env var).

---

## Quick start

### 1 — Clone / copy the project

```bash
git clone <your-repo-url>
cd clothing_review_project
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:** Replace `faiss-cpu` with `faiss-gpu` in `requirements.txt`
> if you have a CUDA-capable GPU.

### 4 — Obtain the dataset

**Option A — Kaggle CLI (automatic download)**

1. Create a Kaggle account and download your API token from
   `https://www.kaggle.com/settings` → *API* → *Create New Token*.
2. Place `kaggle.json` in `~/.kaggle/` (Linux/macOS) or
   `C:\Users\<you>\.kaggle\` (Windows) and set permissions:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Pass `--download` when running the pipeline (see §5 below).

**Option B — Manual download**

Download `Womens Clothing E-Commerce Reviews.csv` from Kaggle and place it
in the project root (or set `DATA_PATH` to its absolute path).

### 5 — Set environment variables

```bash
# Required only for the RAG / Gemini stage
export GEMINI_API_KEY="your-gemini-api-key"

# Optional overrides
export DATA_PATH="/path/to/Womens Clothing E-Commerce Reviews.csv"
export OUTPUT_DIR="/path/to/outputs"
```

> **Security:** Never hard-code API keys in source files.
> The pipeline reads `GEMINI_API_KEY` from the environment only.

### 6 — Run the pipeline

```bash
# Full pipeline (all 5 stages)
python main.py

# Download dataset from Kaggle first, then run
python main.py --download

# Skip deep learning (faster iteration / no GPU)
python main.py --skip-dl

# Skip RAG / Gemini (no API key needed)
python main.py --skip-rag

# ML only — no DL, no RAG
python main.py --skip-dl --skip-rag
```

---

## Pipeline stages

| Stage | Module | What it does |
|-------|--------|--------------|
| 1 | `data_loader.py` | Load CSV, print EDA summary, save `eda_plots.png` |
| 2 | `preprocessing.py` | Clean data, engineer features, label-encode, scale, SMOTE, save CSV splits |
| 3 | `ml_pipeline.py` | Train LR / RF / GB, GridSearchCV tuning, save `ml_comparison.png`, `feature_importance.png`, `best_ml_model.pkl` |
| 4 | `dl_pipeline.py` | Train MLP (TF-IDF) + 1D-CNN (embeddings), save `dl_training_plots.png`, `mlp_sentiment_model.keras`, `cnn_sentiment_model.keras` |
| 5 | `rag_pipeline.py` | Build FAISS index, save `reviews.index` + `metadata.json`, query Gemini |

---

## Configuration

All knobs are in `config.py`. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_PATH` | `"Womens Clothing E-Commerce Reviews.csv"` | CSV location (overridden by env var) |
| `OUTPUT_DIR` | `"outputs"` | Directory for all saved artefacts |
| `RANDOM_STATE` | `42` | Global random seed |
| `TEST_SIZE` | `0.20` | Train / test split ratio |
| `RF_N_ESTIMATORS` | `200` | Random Forest tree count |
| `VOCAB_SIZE` | `20 000` | CNN vocabulary size |
| `MAX_SEQ_LEN` | `150` | CNN sequence length |
| `DL_EPOCHS` | `30` | Max training epochs (EarlyStopping active) |
| `GEMINI_MODEL` | `"gemini-2.5-flash"` | Gemini model identifier |
| `TOP_K` | `8` | Number of reviews retrieved per RAG query |

---

## Bugs fixed vs original notebook

| # | Bug | Fix |
|---|-----|-----|
| 1 | **Hardcoded Gemini API key** in source file | Moved to `GEMINI_API_KEY` environment variable |
| 2 | **Duplicate `train_test_split`** call | Removed redundant second block |
| 3 | **Single `LabelEncoder` reused** across columns — only last column's mapping retained | One `LabelEncoder` instance per column, stored in a dict |
| 4 | **`display()` used without import** — crashes outside Jupyter | Replaced with `print(...to_string())` |
| 5 | **`.h5` model save** — deprecated in TF 2.13+ | Changed to `.keras` native format |
| 6 | **`input_length` in Embedding layer** — deprecated in Keras ≥ 2.13 | Removed; shape inferred from `Input()` layer |
| 7 | **Hardcoded Kaggle CSV path** (`/kaggle/input/...`) — breaks locally | Configurable via `DATA_PATH` env var / `config.py` |
| 8 | **Wrong demo query** (men's T-shirts on a women's dataset) | Replaced with relevant women's clothing queries |
| 9 | **`!pip install` inside notebook** | All dependencies moved to `requirements.txt` |

---

## Outputs

After a full run the `outputs/` directory contains:

```
outputs/
├── eda_plots.png               # EDA visualisations
├── processed_reviews.csv       # Cleaned + feature-engineered data
├── X_train.csv / X_test.csv    # Scaled ML feature splits
├── y_train.csv / y_test.csv    # Target splits
├── ml_comparison.png           # ML metrics / ROC / confusion matrix
├── feature_importance.png      # RF feature importances
├── best_ml_model.pkl           # Best sklearn model (joblib)
├── dl_training_plots.png       # DL training curves + confusion matrices
├── mlp_sentiment_model.keras   # Trained MLP model
├── cnn_sentiment_model.keras   # Trained CNN model
├── reviews.index               # FAISS vector index
└── metadata.json               # Review metadata for RAG
```

---

## Requirements

- Python 3.9 – 3.11
- All packages listed in `requirements.txt`
- Gemini API key (Stage 5 only)
- Kaggle credentials (only if using `--download`)

---

## License

This project is released for educational purposes.
The dataset is provided by [nicapotato on Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) under the CC0 Public Domain licence.
