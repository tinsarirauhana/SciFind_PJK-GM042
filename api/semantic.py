from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import os
import time

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:5173",
            "http://localhost:5000",
            "https://uas-pi-sci-find.vercel.app"
        ]
    }
})

# Root project (satu level di atas folder api/)
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
#  LAZY LOAD — dipanggil saat request pertama masuk
#  Supaya cold start Vercel tidak timeout
# ============================================================

_ready = False
_error = None
_model = None
_embedding_matrix_norm = None
_doc_ids = None
_raw_embeddings = None
_title_to_doc = None
_clean_docs = None


def _load():
    global _ready, _error, _model
    global _embedding_matrix_norm, _doc_ids, _raw_embeddings
    global _title_to_doc, _clean_docs

    if _ready or _error:
        return

    try:
        from sentence_transformers import SentenceTransformer

        # Load clean_documents.json
        with open(os.path.join(SCRIPT_DIR, "clean_documents.json"), "r", encoding="utf-8") as f:
            _clean_docs = json.load(f)

        # Load embeddings.json
        with open(os.path.join(SCRIPT_DIR, "embeddings.json"), "r", encoding="utf-8") as f:
            _raw_embeddings = json.load(f)

        _title_to_doc = {doc["title"]: doc for doc in _clean_docs}

        # Susun matrix embedding + normalisasi
        _doc_ids = list(_raw_embeddings.keys())
        emb_matrix = np.array([_raw_embeddings[d]["embedding"] for d in _doc_ids])
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        _embedding_matrix_norm = emb_matrix / norms

        # Load model sentence-transformers
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        _ready = True
        print("semantic.py: model & embeddings loaded OK")

    except Exception as exc:
        _error = str(exc)
        print(f"semantic.py: LOAD FAILED — {exc}")


def _resolve_doc(doc_id):
    """Cari dokumen dari title_to_doc, fallback ke loop jika perlu."""
    title = doc_id.replace("_", " ")
    doc = _title_to_doc.get(title)
    if doc is None:
        for d in _clean_docs:
            if d["title"].replace(" ", "_") == doc_id:
                return d
    return doc


# ============================================================
#  ENDPOINT: /api/semantic-search
# ============================================================

@app.route("/api/semantic-search", methods=["GET", "POST"])
def api_semantic_search():
    """
    Semantic search berdasarkan makna query.
    GET  /api/semantic-search?query=time+travel&top_k=10
    POST /api/semantic-search  {"query": "time travel", "top_k": 10}
    """
    if request.method == "POST":
        data = request.get_json() or {}
        query = data.get("query", "")
        top_k = int(data.get("top_k", 10))
    else:
        query = request.args.get("query", "")
        top_k = int(request.args.get("top_k", 10))

    if not query:
        return jsonify({"error": "query is required"}), 400

    _load()
    if _error:
        return jsonify({"error": f"Gagal load model: {_error}"}), 500

    try:
        t0 = time.time()

        query_emb = _model.encode(query, convert_to_numpy=True)
        norm = np.linalg.norm(query_emb)
        if norm == 0:
            return jsonify({"query": query, "results": [], "latency_ms": 0})

        query_norm = query_emb / norm
        similarities = _embedding_matrix_norm @ query_norm
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score <= 0:
                continue
            doc = _resolve_doc(_doc_ids[idx])
            if doc:
                results.append({
                    "title": doc["title"],
                    "poster": doc.get("poster", ""),
                    "description": doc.get("description", ""),
                    "semantic_score": round(score, 4),
                })

        return jsonify({
            "query": query,
            "results": results,
            "latency_ms": round((time.time() - t0) * 1000, 2),
        })

    except Exception as e:
        print(f"Error /api/semantic-search: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
#  ENDPOINT: /api/recommend
# ============================================================

@app.route("/api/recommend", methods=["GET", "POST"])
def api_recommend():
    """
    Content-based recommendation berdasarkan judul film.
    GET  /api/recommend?title=Dune&top_n=6
    POST /api/recommend  {"title": "Dune", "top_n": 6}
    """
    if request.method == "POST":
        data = request.get_json() or {}
        title = data.get("title", "")
        top_n = int(data.get("top_n", 6))
    else:
        title = request.args.get("title", "")
        top_n = int(request.args.get("top_n", 6))

    if not title:
        return jsonify({"error": "title is required"}), 400

    _load()
    if _error:
        return jsonify({"error": f"Gagal load model: {_error}"}), 500

    try:
        t0 = time.time()

        # Resolve doc_id (case-insensitive)
        doc_id = title.replace(" ", "_")
        if doc_id not in _raw_embeddings:
            matched = next(
                (k for k in _raw_embeddings if k.lower() == doc_id.lower()), None
            )
            if matched is None:
                return jsonify({"error": f"Film '{title}' tidak ditemukan di embeddings."}), 404
            doc_id = matched

        target_idx = _doc_ids.index(doc_id)
        target_vec = _embedding_matrix_norm[target_idx]

        similarities = _embedding_matrix_norm @ target_vec
        similarities[target_idx] = -1  # exclude film itu sendiri

        top_indices = np.argsort(similarities)[::-1][:top_n]

        recommendations = []
        for idx in top_indices:
            score = float(similarities[idx])
            doc = _resolve_doc(_doc_ids[idx])
            if doc:
                recommendations.append({
                    "title": doc["title"],
                    "poster": doc.get("poster", ""),
                    "description": doc.get("description", ""),
                    "similarity_score": round(score, 4),
                })

        return jsonify({
            "title": title,
            "recommendations": recommendations,
            "latency_ms": round((time.time() - t0) * 1000, 2),
        })

    except Exception as e:
        print(f"Error /api/recommend: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
#  JALANKAN LOKAL (opsional, port berbeda dari index.py)
# ============================================================

if __name__ == "__main__":
    print("semantic.py running on http://localhost:5001")
    app.run(debug=True, port=5001)
