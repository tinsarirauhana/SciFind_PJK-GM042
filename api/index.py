from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from difflib import get_close_matches
from collections import Counter
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

# ============================================================
#  PATH RESOLUTION — works both locally and on Vercel
# ============================================================

# api/index.py is at <root>/api/index.py → parent is <root>
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
#  LOAD CORE DATA (TF-IDF + CLEAN DOCS)
# ============================================================

print("Loading core data...")
with open(os.path.join(SCRIPT_DIR, "clean_documents.json"), "r", encoding="utf-8") as f:
    CLEAN_DOCS = json.load(f)

with open(os.path.join(SCRIPT_DIR, "tfidf_index.json"), "r", encoding="utf-8") as f:
    TFIDF = json.load(f)

tfidf_matrix = np.array(TFIDF["matrix"])
vocab = TFIDF["vocab"]
vocab_index = {w: i for i, w in enumerate(vocab)}
idf = np.array(TFIDF["idf"])
filenames = TFIDF["filenames"]

vocab_set = set(vocab)
for doc in CLEAN_DOCS:
    if "tokens" in doc:
        vocab_set.update(doc["tokens"])
    if "title" in doc:
        vocab_set.update(doc["title"].lower().replace("-", " ").replace(":", " ").split())

print(f"  Loaded {len(CLEAN_DOCS)} documents, vocab size {len(vocab_set)}")

# ============================================================
#  LAZY LOAD — SEMANTIC / EMBEDDINGS
#  sentence-transformers is large; load once on first request
#  so Vercel cold-start for /api/search stays fast.
# ============================================================

_semantic_ready = False
_semantic_error = None
_sem_model = None
_embedding_matrix_norm = None
_sem_doc_ids = None
_raw_embeddings = None
_title_to_doc = None


def _load_semantic():
    global _semantic_ready, _semantic_error
    global _sem_model, _embedding_matrix_norm, _sem_doc_ids, _raw_embeddings, _title_to_doc

    if _semantic_ready or _semantic_error:
        return

    try:
        from sentence_transformers import SentenceTransformer

        embeddings_path = os.path.join(SCRIPT_DIR, "embeddings.json")
        with open(embeddings_path, "r", encoding="utf-8") as f:
            _raw_embeddings = json.load(f)

        _title_to_doc = {doc["title"]: doc for doc in CLEAN_DOCS}

        _sem_doc_ids = list(_raw_embeddings.keys())
        emb_matrix = np.array([_raw_embeddings[d]["embedding"] for d in _sem_doc_ids])

        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        _embedding_matrix_norm = emb_matrix / norms

        _sem_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        _semantic_ready = True
        print("  Semantic module loaded OK")

    except Exception as exc:
        _semantic_error = str(exc)
        print(f"  Semantic load FAILED: {exc}")


# ============================================================
#  PREPROCESSING UTILS
# ============================================================

title_map = {d["title"]: d for d in CLEAN_DOCS}


def preprocess_query(q):
    return q.lower().replace("-", " ").split()


def compute_tf(tokens):
    vec = np.zeros(len(vocab_index))
    count = Counter(tokens)
    for word, freq in count.items():
        if word in vocab_index:
            vec[vocab_index[word]] = freq
    return vec


def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def autocorrect_query(query):
    words = query.lower().split()
    corrected, corrections = [], []
    for word in words:
        if len(word) <= 2 or word in vocab_set:
            corrected.append(word)
        else:
            matches = get_close_matches(word, vocab_set, n=1, cutoff=0.5)
            if matches:
                corrected.append(matches[0])
                corrections.append({"original": word, "corrected": matches[0]})
            else:
                corrected.append(word)
    return " ".join(corrected), corrections


def jaccard_similarity(q_tokens, doc_tokens, title_tokens):
    if not q_tokens:
        return 0.0
    q_set = set(q_tokens)
    inter = len(q_set & set(doc_tokens))
    union = len(q_set | set(doc_tokens))
    base = inter / union if union > 0 else 0.0
    if q_set & set(title_tokens):
        base += 0.3
    return base


# ============================================================
#  EVALUATION HELPERS
# ============================================================

def evaluate_search(query, results, method, runtime_ms):
    if not results:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0,
                "accuracy": round(runtime_ms, 2), "query": query, "method": method}

    score_key = {"tfidf": "tfidf_score", "jaccard": "jaccard_score"}.get(method, "score")
    scores = [r[score_key] for r in results[:10] if score_key in r]

    if not scores:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0,
                "accuracy": round(runtime_ms, 2), "query": query, "method": method}

    precision = sum(scores[:5]) / len(scores[:5])
    recall = sum(scores) / len(scores)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1_score": round(f1, 4), "accuracy": round(runtime_ms, 2),
            "query": query, "method": method}


# ============================================================
#  CORE SEARCH FUNCTION
# ============================================================

def search(query, method="hybrid", top_k=10):
    t0 = time.time()
    corrected_query, corrections = autocorrect_query(query)
    q_tokens = preprocess_query(corrected_query)

    # TF-IDF
    q_vec = compute_tf(q_tokens) * idf
    tfidf_scores = []
    for i, doc_vec in enumerate(tfidf_matrix):
        sim = cosine_similarity(q_vec, doc_vec)
        title_tokens = filenames[i].lower().replace("-", " ").split()
        if set(q_tokens) & set(title_tokens):
            sim += 0.3
        tfidf_scores.append((i, sim))
    tfidf_ranked = sorted([(i, s) for i, s in tfidf_scores if s > 0], key=lambda x: x[1], reverse=True)

    # Jaccard
    jaccard_scores = []
    for doc in CLEAN_DOCS:
        sim = jaccard_similarity(q_tokens, doc["tokens"],
                                 doc["title"].lower().replace("-", " ").split())
        jaccard_scores.append((doc["title"], sim))
    jaccard_ranked = sorted([(t, s) for t, s in jaccard_scores if s > 0], key=lambda x: x[1], reverse=True)

    result_tfidf = []
    for idx, score in tfidf_ranked[:top_k]:
        title = filenames[idx]
        if title in title_map:
            d = title_map[title]
            result_tfidf.append({"title": d["title"], "poster": d["poster"],
                                  "description": d["description"], "tfidf_score": float(score)})

    result_jaccard = []
    for title, score in jaccard_ranked[:top_k]:
        if title in title_map:
            d = title_map[title]
            result_jaccard.append({"title": d["title"], "poster": d["poster"],
                                    "description": d["description"], "jaccard_score": float(score)})

    runtime_ms = (time.time() - t0) * 1000
    evaluation = {
        "tfidf": evaluate_search(corrected_query, result_tfidf[:top_k], "tfidf", runtime_ms),
        "jaccard": evaluate_search(corrected_query, result_jaccard[:top_k], "jaccard", runtime_ms),
    }

    if method == "hybrid":
        combined = {}
        for item in result_tfidf:
            combined[item["title"]] = {"data": item, "tfidf_score": item["tfidf_score"], "jaccard_score": 0}
        for item in result_jaccard:
            t = item["title"]
            if t in combined:
                combined[t]["jaccard_score"] = item["jaccard_score"]
            else:
                combined[t] = {"data": item, "tfidf_score": 0, "jaccard_score": item["jaccard_score"]}

        hybrid_results = []
        for t, s in combined.items():
            r = s["data"].copy()
            r["score"] = float(s["tfidf_score"] * 0.7 + s["jaccard_score"] * 0.3)
            r["tfidf_score"] = float(s["tfidf_score"])
            r["jaccard_score"] = float(s["jaccard_score"])
            hybrid_results.append(r)
        hybrid_results.sort(key=lambda x: x["score"], reverse=True)
        results = hybrid_results[:top_k]
    elif method == "tfidf":
        results = result_tfidf
    else:
        results = result_jaccard

    return {"results": results, "corrected_query": corrected_query,
            "corrections": corrections, "total": len(results), "evaluation": evaluation}


# ============================================================
#  API ENDPOINTS
# ============================================================

@app.route("/api/search", methods=["GET", "POST"])
def api_search():
    if request.method == "POST":
        data = request.get_json() or {}
        query = data.get("query", "")
        method = data.get("method", "hybrid")
        top_k = int(data.get("top_k", 10))
    else:
        query = request.args.get("query", "")
        method = request.args.get("method", "hybrid")
        top_k = int(request.args.get("top_k", 10))

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        result = search(query, method, top_k)
        return jsonify({
            "query": query,
            "corrected_query": result["corrected_query"] if result["corrected_query"] != query else None,
            "corrections": result["corrections"] or None,
            "method": method,
            "results": result["results"],
            "evaluation": result.get("evaluation"),
        })
    except Exception as e:
        print(f"Error /api/search: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/semantic-search", methods=["GET", "POST"])
def api_semantic_search():
    """
    Semantic search menggunakan sentence-transformers.
    GET  /api/semantic-search?query=...&top_k=10
    POST /api/semantic-search  { "query": "...", "top_k": 10 }
    """
    if request.method == "POST":
        data = request.get_json() or {}
        query = data.get("query", "")
        top_k = int(data.get("top_k", 10))
    else:
        query = request.args.get("query", "")
        top_k = int(request.args.get("top_k", 10))

    if not query:
        return jsonify({"error": "Query is required"}), 400

    _load_semantic()

    if _semantic_error:
        return jsonify({"error": f"Semantic module failed to load: {_semantic_error}"}), 500

    try:
        t0 = time.time()

        query_emb = _sem_model.encode(query, convert_to_numpy=True)
        norm = np.linalg.norm(query_emb)
        if norm == 0:
            return jsonify({"results": [], "latency_ms": 0, "query": query})

        query_norm = query_emb / norm
        similarities = _embedding_matrix_norm @ query_norm
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score <= 0:
                continue
            doc_id = _sem_doc_ids[idx]
            title = doc_id.replace("_", " ")
            doc = _title_to_doc.get(title)
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


@app.route("/api/recommend", methods=["GET", "POST"])
def api_recommend():
    """
    Content-based recommendation berdasarkan judul film.
    GET  /api/recommend?title=Dune&top_n=6
    POST /api/recommend  { "title": "Dune", "top_n": 6 }
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

    _load_semantic()

    if _semantic_error:
        return jsonify({"error": f"Semantic module failed to load: {_semantic_error}"}), 500

    try:
        t0 = time.time()

        # Resolve doc_id (case-insensitive and fuzzy match)
        doc_id = title.replace(" ", "_")
        if doc_id not in _raw_embeddings:
            matched = next((k for k in _raw_embeddings if k.lower() == title.lower()), None)
            if matched is None:
                matched = next((k for k in _raw_embeddings if title.lower() in k.lower()), None)
            if matched is None:
                close = get_close_matches(title, list(_raw_embeddings.keys()), n=1, cutoff=0.6)
                matched = close[0] if close else None
            if matched is None:
                return jsonify({"error": f"Film '{title}' tidak ditemukan di embeddings."}), 404
            doc_id = matched

        target_idx = _sem_doc_ids.index(doc_id)
        target_vec = _embedding_matrix_norm[target_idx]

        similarities = _embedding_matrix_norm @ target_vec
        similarities[target_idx] = -1  # exclude self

        top_indices = np.argsort(similarities)[::-1][:top_n]

        recommendations = []
        for idx in top_indices:
            rec_doc_id = _sem_doc_ids[idx]
            score = float(similarities[idx])
            rec_title = rec_doc_id.replace("_", " ")
            doc = _title_to_doc.get(rec_title)
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


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "total_documents": len(CLEAN_DOCS),
        "vocabulary_size": len(vocab_set),
        "semantic_ready": _semantic_ready,
    })


# ============================================================
#  LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SciFind Search Engine API Server")
    print("=" * 60)
    print(f"  Backend ready with {len(CLEAN_DOCS)} sci-fi movies/series")
    print(f"  http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
