"""
rag_pipeline.py
---------------
Retrieval-Augmented Generation (RAG) pipeline:
  1. Build dense embeddings for every review with SentenceTransformer
  2. Index embeddings in FAISS (cosine similarity via inner product on L2-normalised vectors)
  3. Persist the FAISS index and metadata
  4. At query time: embed query → FAISS search → build LLM prompt → Gemini response

Security fix
------------
The original notebook had a raw API key hardcoded in the source:
    GEMINI_API_KEY = "AIzaSy..."
This is a critical security vulnerability — the key is committed to version
control and visible to anyone who reads the file.

FIX: Load the key from the GEMINI_API_KEY environment variable only.
     Set it before running:
         export GEMINI_API_KEY="your-key-here"   # Linux / macOS
         set GEMINI_API_KEY=your-key-here         # Windows cmd
"""

import json
import os
import warnings
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import config

warnings.filterwarnings("ignore")

# Optional heavy imports — graceful failure if packages are missing
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    print("[rag_pipeline] WARNING: faiss-cpu not installed — RAG unavailable.")

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    print("[rag_pipeline] WARNING: sentence-transformers not installed — RAG unavailable.")

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    print("[rag_pipeline] WARNING: google-generativeai not installed — LLM unavailable.")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_rag_data(df: pd.DataFrame):
    """
    Build document strings and metadata records from the DataFrame.

    Each document is a structured text block containing all review fields.
    Metadata is stored separately for display in search results.

    Parameters
    ----------
    df : Cleaned DataFrame (review text must be non-empty).

    Returns
    -------
    documents : List[str]  — one text block per review.
    metadata  : List[Dict] — parallel list of structured metadata.
    """
    # Fill any remaining NaN with empty string for safe string formatting
    df = df.fillna("").reset_index(drop=True)

    # Remove rows where review text is blank after stripping
    df = df[df["Review Text"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)

    documents: List[str] = []
    metadata:  List[Dict[str, Any]] = []

    for i, row in df.iterrows():
        doc = (
            f"Product ID: {row['Clothing ID']}\n"
            f"Age: {row['Age']}\n"
            f"Rating: {row['Rating']}\n"
            f"Recommended: {row['Recommended IND']}\n"
            f"Division: {row['Division Name']}\n"
            f"Department: {row['Department Name']}\n"
            f"Class: {row['Class Name']}\n"
            f"Title: {row['Title']}\n"
            f"Review: {row['Review Text']}"
        )
        documents.append(doc)

        # Safe integer parsing for nullable fields
        age_val = int(row["Age"]) if str(row["Age"]).isdigit() else None

        metadata.append({
            "row_id"      : int(i),
            "clothing_id" : int(row["Clothing ID"]) if str(row["Clothing ID"]).isdigit() else None,
            "age"         : age_val,
            "rating"      : int(row["Rating"]),
            "recommended" : int(row["Recommended IND"]),
            "division"    : str(row["Division Name"]),
            "department"  : str(row["Department Name"]),
            "class_name"  : str(row["Class Name"]),
            "title"       : str(row["Title"]),
        })

    print(f"[rag_pipeline] Prepared {len(documents):,} documents")
    return documents, metadata


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def build_embeddings(documents: List[str], model_name: str = config.EMBED_MODEL_NAME) -> np.ndarray:
    """
    Encode all documents with a SentenceTransformer model.

    Parameters
    ----------
    documents  : List of raw text strings to encode.
    model_name : HuggingFace model identifier.

    Returns
    -------
    np.ndarray of shape (N, D) in float32.
    """
    if not _ST_AVAILABLE:
        raise RuntimeError("sentence-transformers is not installed.")

    embedder   = SentenceTransformer(model_name)
    embeddings = embedder.encode(
        documents,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    print(f"[rag_pipeline] Embeddings shape: {embeddings.shape}")
    return embeddings, embedder


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray) -> "faiss.IndexFlatIP":
    """
    Normalise embeddings and build a FAISS inner-product index.
    After L2 normalisation inner-product == cosine similarity.

    Parameters
    ----------
    embeddings : float32 array of shape (N, D).

    Returns
    -------
    Populated FAISS IndexFlatIP.
    """
    if not _FAISS_AVAILABLE:
        raise RuntimeError("faiss-cpu is not installed.")

    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index     = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"[rag_pipeline] FAISS index built — {index.ntotal:,} vectors (dim={dimension})")
    return index


def save_rag_artifacts(
    index,
    metadata: List[Dict],
    index_path: str = config.FAISS_INDEX_PATH,
    meta_path:  str = config.METADATA_PATH,
) -> None:
    """Persist the FAISS index and metadata JSON to disk."""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f)
    print(f"[rag_pipeline] FAISS index saved  → {index_path}")
    print(f"[rag_pipeline] Metadata saved      → {meta_path}")


def load_rag_artifacts(
    index_path: str = config.FAISS_INDEX_PATH,
    meta_path:  str = config.METADATA_PATH,
):
    """Reload a previously saved FAISS index and metadata JSON."""
    if not _FAISS_AVAILABLE:
        raise RuntimeError("faiss-cpu is not installed.")
    index    = faiss.read_index(index_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    print(f"[rag_pipeline] Loaded index with {index.ntotal:,} vectors")
    return index, metadata


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_reviews(
    query: str,
    embedder,
    index,
    documents:  List[str],
    metadata:   List[Dict],
    top_k:      int = config.TOP_K,
) -> List[Dict]:
    """
    Retrieve the top-k most relevant reviews for a natural-language query.

    Parameters
    ----------
    query     : User's question or request.
    embedder  : Fitted SentenceTransformer.
    index     : Populated FAISS index.
    documents : Parallel list of document strings.
    metadata  : Parallel list of metadata dicts.
    top_k     : Number of results to return.

    Returns
    -------
    List of result dicts with keys: score, text, meta.
    """
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    scores, ids = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append({
            "score": float(score),
            "text" : documents[idx],
            "meta" : metadata[idx],
        })
    return results


# ---------------------------------------------------------------------------
# RAG prompt + LLM call
# ---------------------------------------------------------------------------

def setup_gemini() -> "genai.GenerativeModel":
    """
    Configure the Gemini API client using the environment variable.

    Raises
    ------
    EnvironmentError if GEMINI_API_KEY is not set.
    RuntimeError    if google-generativeai is not installed.
    """
    if not _GENAI_AVAILABLE:
        raise RuntimeError("google-generativeai is not installed.")

    api_key = config.GEMINI_API_KEY
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Run:  export GEMINI_API_KEY='your-key-here'  (Linux/macOS)\n"
            "  or  set GEMINI_API_KEY=your-key-here       (Windows)"
        )

    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(config.GEMINI_MODEL)
    print(f"[rag_pipeline] Gemini model '{config.GEMINI_MODEL}' ready")
    return llm


def build_prompt(user_query: str, retrieved: List[Dict]) -> str:
    """
    Construct a RAG prompt from the user query and retrieved review context.

    Parameters
    ----------
    user_query : The customer's natural-language question.
    retrieved  : List of retrieved result dicts from search_reviews().

    Returns
    -------
    Formatted prompt string.
    """
    context_blocks = []
    for i, r in enumerate(retrieved):
        block = (
            f"Result {i + 1}\n"
            f"Similarity Score : {r['score']:.3f}\n"
            f"Product ID       : {r['meta']['clothing_id']}\n"
            f"Age              : {r['meta']['age']}\n"
            f"Rating           : {r['meta']['rating']}\n"
            f"Department       : {r['meta']['department']}\n"
            f"Class            : {r['meta']['class_name']}\n"
            f"Title            : {r['meta']['title']}\n"
            f"Review:\n{r['text']}"
        )
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    prompt = (
        "You are an expert AI shopping assistant for women's fashion.\n"
        "Use ONLY the retrieved customer review data below.\n\n"
        "TASK:\n"
        "Help the user with product recommendations using real reviews.\n\n"
        f"USER REQUEST:\n{user_query}\n\n"
        f"RETRIEVED REVIEWS:\n{context}\n\n"
        "Instructions:\n"
        "1. Recommend the best matching products.\n"
        "2. Mention Product IDs.\n"
        "3. Comment on fit, comfort, quality, and style.\n"
        "4. Warn the user if many reviews flag issues.\n"
        "5. Be concise and useful.\n"
    )
    return prompt


def ask_fashion_assistant(
    user_query: str,
    llm,
    embedder,
    index,
    documents: List[str],
    metadata:  List[Dict],
) -> str:
    """
    Full RAG pipeline: search → prompt → generate → return answer text.

    Parameters
    ----------
    user_query : Natural-language question from the user.
    llm        : Configured Gemini GenerativeModel.
    embedder   : Fitted SentenceTransformer.
    index      : Populated FAISS index.
    documents  : Document strings.
    metadata   : Metadata dicts.

    Returns
    -------
    str : Generated answer from Gemini.
    """
    retrieved = search_reviews(user_query, embedder, index, documents, metadata)
    prompt    = build_prompt(user_query, retrieved)
    response  = llm.generate_content(prompt)
    return response.text
