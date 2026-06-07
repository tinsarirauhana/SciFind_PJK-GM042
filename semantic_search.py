"""
semantic_search.py

Modul untuk:
1. Semantic Search  - cari dokumen berdasarkan makna query
2. Content-Based Filtering - rekomendasi film serupa berdasarkan embedding

Digunakan oleh app.py sebagai modul tambahan.
"""

import os
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================
#  KONFIGURASI
# ============================================================

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(SCRIPT_DIR, "embeddings.json")
CLEAN_DOCS_FILE = os.path.join(SCRIPT_DIR, "clean_documents.json")

# ============================================================
#  LOAD MODEL DAN DATA
# ============================================================

print("SemanticSearch: Memuat model...")
model = SentenceTransformer(MODEL_NAME)
print("SemanticSearch: Model berhasil dimuat.")

print("SemanticSearch: Memuat embeddings.json...")
with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    raw_embeddings = json.load(f)

print("SemanticSearch: Memuat clean_documents.json...")
with open(CLEAN_DOCS_FILE, "r", encoding="utf-8") as f:
    CLEAN_DOCS = json.load(f)

# Buat mapping title -> doc
title_to_doc = {doc["title"]: doc for doc in CLEAN_DOCS}

# Susun embeddings menjadi matrix numpy dan list doc_ids
doc_ids = list(raw_embeddings.keys())
embedding_matrix = np.array([raw_embeddings[doc_id]["embedding"] for doc_id in doc_ids])

# Normalisasi embedding matrix untuk cosine similarity yang efisien
norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
norms[norms == 0] = 1
embedding_matrix_normalized = embedding_matrix / norms

print(f"SemanticSearch: {len(doc_ids)} embeddings siap digunakan.")

# ============================================================
#  HELPER: doc_id ke title
# ============================================================

def doc_id_to_title(doc_id):
    return doc_id.replace("_", " ")

def title_to_doc_id(title):
    return title.replace(" ", "_")

# ============================================================
#  SEMANTIC SEARCH
# ============================================================

def semantic_search(query, top_k=10):
    """
    Cari dokumen yang paling relevan berdasarkan makna semantik query.

    Parameter:
        query  : string query dari pengguna
        top_k  : jumlah hasil yang dikembalikan

    Return:
        dict berisi results dan metadata evaluasi
    """
    start_time = time.time()

    query_embedding = model.encode(query, convert_to_numpy=True)
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return {"results": [], "latency_ms": 0}

    query_normalized = query_embedding / query_norm

    similarities = embedding_matrix_normalized @ query_normalized

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        doc_id = doc_ids[idx]
        score = float(similarities[idx])

        if score <= 0:
            continue

        title = doc_id_to_title(doc_id)
        doc = title_to_doc.get(title)

        if doc is None:
            for d in CLEAN_DOCS:
                if d["title"].replace(" ", "_") == doc_id:
                    doc = d
                    break

        if doc:
            results.append({
                "title": doc["title"],
                "poster": doc.get("poster", ""),
                "description": doc.get("description", ""),
                "semantic_score": round(score, 4)
            })

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "results": results,
        "latency_ms": latency_ms
    }

# ============================================================
#  CONTENT-BASED FILTERING (REKOMENDASI FILM SERUPA)
# ============================================================

def get_recommendations(title, top_n=6):
    """
    Rekomendasikan film serupa berdasarkan cosine similarity embedding.

    Parameter:
        title  : judul film yang sedang dibuka pengguna
        top_n  : jumlah rekomendasi yang dikembalikan (tidak termasuk film itu sendiri)

    Return:
        list of dict berisi film yang direkomendasikan
    """
    start_time = time.time()

    doc_id = title_to_doc_id(title)

    if doc_id not in raw_embeddings:
        for key in raw_embeddings.keys():
            if key.lower() == doc_id.lower():
                doc_id = key
                break
        else:
            return {"results": [], "error": f"Film '{title}' tidak ditemukan di embeddings."}

    target_idx = doc_ids.index(doc_id)
    target_vector = embedding_matrix_normalized[target_idx]

    similarities = embedding_matrix_normalized @ target_vector

    similarities[target_idx] = -1

    top_indices = np.argsort(similarities)[::-1][:top_n]

    recommendations = []
    for idx in top_indices:
        rec_doc_id = doc_ids[idx]
        score = float(similarities[idx])

        rec_title = doc_id_to_title(rec_doc_id)
        doc = title_to_doc.get(rec_title)

        if doc is None:
            for d in CLEAN_DOCS:
                if d["title"].replace(" ", "_") == rec_doc_id:
                    doc = d
                    break

        if doc:
            recommendations.append({
                "title": doc["title"],
                "poster": doc.get("poster", ""),
                "description": doc.get("description", ""),
                "similarity_score": round(score, 4)
            })

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "results": recommendations,
        "latency_ms": latency_ms
    }
