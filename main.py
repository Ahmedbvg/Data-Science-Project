"""
main.py
-------
End-to-end orchestrator for the Women's E-Commerce Clothing Reviews project.

Runs the four pipeline stages in order:
  Stage 1 – Data collection & EDA
  Stage 2 – Preprocessing (cleaning, features, encoding, SMOTE)
  Stage 3 – Classical ML (Logistic Regression, Random Forest, Gradient Boosting)
  Stage 4 – Deep Learning (MLP on TF-IDF, 1D-CNN on embeddings)
  Stage 5 – RAG + LLM (FAISS + Gemini)  [optional — requires GEMINI_API_KEY]

Usage
-----
    python main.py                      # run all stages
    python main.py --skip-dl            # skip deep learning (faster)
    python main.py --skip-rag           # skip RAG / Gemini
    python main.py --skip-dl --skip-rag # ML only

Environment variables
---------------------
    DATA_PATH        – override default CSV path
    OUTPUT_DIR       – override default output directory (default: "outputs")
    GEMINI_API_KEY   – required for Stage 5 (RAG)
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import config

# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Women's Clothing Reviews — full ML/DL/RAG pipeline"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download dataset from Kaggle before running (requires kaggle.json)",
    )
    parser.add_argument(
        "--skip-dl", action="store_true",
        help="Skip the deep-learning stage (MLP + CNN)",
    )
    parser.add_argument(
        "--skip-rag", action="store_true",
        help="Skip the RAG / Gemini stage",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Stage 1 – Data
# ---------------------------------------------------------------------------

def stage_data(download: bool = False):
    """Load the dataset and run EDA."""
    from data_loader import download_from_kaggle, load_data, print_eda_summary, plot_eda

    print("\n" + "=" * 60)
    print("  STAGE 1 — Data Collection & EDA")
    print("=" * 60)

    if download:
        download_from_kaggle()

    df = load_data()
    print_eda_summary(df)
    plot_eda(df)
    return df


# ---------------------------------------------------------------------------
# Stage 2 – Preprocessing
# ---------------------------------------------------------------------------

def stage_preprocessing(df):
    """Clean data, engineer features, encode, scale, split, SMOTE."""
    from preprocessing import (
        clean_data,
        engineer_features,
        encode_and_scale,
        split_data,
        apply_smote,
        compute_class_weights,
        save_splits,
    )

    print("\n" + "=" * 60)
    print("  STAGE 2 — Preprocessing")
    print("=" * 60)

    df_clean = clean_data(df)
    df_feat  = engineer_features(df_clean)

    X_scaled, y, scaler, encoders = encode_and_scale(df_feat)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # SMOTE + class weights (for reference / use with weighted models)
    X_resampled, y_resampled = apply_smote(X_train, y_train)
    class_weights            = compute_class_weights(y_train)

    save_splits(df_clean, X_train, X_test, y_train, y_test)

    return df_clean, X_train, X_test, y_train, y_test, class_weights


# ---------------------------------------------------------------------------
# Stage 3 – Classical ML
# ---------------------------------------------------------------------------

def stage_ml():
    """Train, tune, and compare sklearn models."""
    from ml_pipeline import (
        load_ml_splits,
        run_logistic_regression,
        run_random_forest,
        run_gradient_boosting,
        tune_random_forest,
        plot_model_comparison,
        plot_feature_importance,
        save_best_model,
    )

    print("\n" + "=" * 60)
    print("  STAGE 3 — Machine Learning")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_ml_splits()

    lr_results = run_logistic_regression(X_train, X_test, y_train, y_test)
    rf_results = run_random_forest(X_train, X_test, y_train, y_test)
    gb_results = run_gradient_boosting(X_train, X_test, y_train, y_test)
    best_rf_results = tune_random_forest(X_train, X_test, y_train, y_test)

    all_results = [lr_results, rf_results, gb_results, best_rf_results]
    plot_model_comparison(all_results, y_test)

    # Feature importance for the tuned RF
    best_model = best_rf_results["model"]
    plot_feature_importance(best_model, X_train.columns)
    save_best_model(best_model)

    return all_results


# ---------------------------------------------------------------------------
# Stage 4 – Deep Learning
# ---------------------------------------------------------------------------

def stage_dl(df_clean):
    """Train MLP (TF-IDF) and 1D-CNN (embeddings) sentiment classifiers."""
    from dl_pipeline import (
        prepare_text_data,
        build_tfidf_features,
        train_mlp,
        build_sequence_features,
        train_cnn,
        evaluate_dl_model,
        plot_dl_results,
        demo_inference,
        save_dl_models,
    )

    print("\n" + "=" * 60)
    print("  STAGE 4 — Deep Learning")
    print("=" * 60)

    X_tr, X_te, y_tr, y_te = prepare_text_data(df_clean)

    # MLP on TF-IDF
    X_tr_tfidf, X_te_tfidf, tfidf     = build_tfidf_features(X_tr, X_te)
    mlp_model, history_mlp             = train_mlp(X_tr_tfidf, y_tr)
    mlp_results                        = evaluate_dl_model("MLP (TF-IDF)", mlp_model, X_te_tfidf, y_te)

    # CNN on word embeddings
    X_tr_seq, X_te_seq, tokenizer      = build_sequence_features(X_tr, X_te)
    cnn_model, history_cnn             = train_cnn(X_tr_seq, y_tr)
    cnn_results                        = evaluate_dl_model("CNN (Embeddings)", cnn_model, X_te_seq, y_te)

    plot_dl_results(history_mlp, history_cnn, mlp_results, cnn_results, y_te)
    demo_inference(cnn_model, tokenizer)
    save_dl_models(mlp_model, cnn_model)


# ---------------------------------------------------------------------------
# Stage 5 – RAG + LLM
# ---------------------------------------------------------------------------

def stage_rag(df_clean):
    """Build FAISS index and run example RAG queries via Gemini."""
    from rag_pipeline import (
        prepare_rag_data,
        build_embeddings,
        build_faiss_index,
        save_rag_artifacts,
        setup_gemini,
        ask_fashion_assistant,
    )

    print("\n" + "=" * 60)
    print("  STAGE 5 — RAG + LLM (Gemini)")
    print("=" * 60)

    documents, metadata        = prepare_rag_data(df_clean)
    embeddings, embedder       = build_embeddings(documents)
    index                      = build_faiss_index(embeddings)
    save_rag_artifacts(index, metadata)

    try:
        llm = setup_gemini()
    except EnvironmentError as e:
        print(f"\n[main] Skipping LLM queries: {e}")
        return

    # Example queries — all relevant to the women's clothing dataset
    queries = [
        "I'm looking for a comfortable summer dress. What do customers say about fit and fabric?",
        "Are there any tops customers love for being true to size?",
        "Which clothing items have the most complaints about quality?",
    ]

    for q in queries:
        print(f"\n{'─' * 60}")
        print(f"QUERY : {q}")
        print("─" * 60)
        answer = ask_fashion_assistant(q, llm, embedder, index, documents, metadata)
        print(answer)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Stage 1 – data
    df = stage_data(download=args.download)

    # Stage 2 – preprocessing
    df_clean, X_train, X_test, y_train, y_test, class_weights = stage_preprocessing(df)

    # Stage 3 – classical ML
    stage_ml()

    # Stage 4 – deep learning (optional)
    if not args.skip_dl:
        stage_dl(df_clean)
    else:
        print("\n[main] Deep-learning stage skipped (--skip-dl).")

    # Stage 5 – RAG / LLM (optional)
    if not args.skip_rag:
        stage_rag(df_clean)
    else:
        print("\n[main] RAG stage skipped (--skip-rag).")

    print("\n" + "=" * 60)
    print("  Pipeline complete — outputs written to:", config.OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
