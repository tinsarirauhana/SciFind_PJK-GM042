from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from difflib import get_close_matches
from collections import Counter
import os

app = Flask(__name__)
# Allow CORS for frontend domains (explicit vercel frontend domain)
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:5173",
            "http://localhost:5000",
            "https://uas-pi-sci-find.vercel.app"
        ]
    }
})

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
#  LOAD CLEAN DOCUMENTS & TF-IDF INDEX
# ============================================================

print("Loading data...")
clean_docs_path = os.path.join(SCRIPT_DIR, "clean_documents.json")
tfidf_index_path = os.path.join(SCRIPT_DIR, "tfidf_index.json")

with open(clean_docs_path, "r", encoding='utf-8') as f:
    CLEAN_DOCS = json.load(f)

with open(tfidf_index_path, "r", encoding='utf-8') as f:
    TFIDF = json.load(f)

tfidf_matrix = np.array(TFIDF["matrix"])
vocab = TFIDF["vocab"]
vocab_index = {w: i for i, w in enumerate(vocab)}
idf = np.array(TFIDF["idf"])
filenames = TFIDF["filenames"]

# Build vocabulary set for auto-correct
vocab_set = set(vocab)
for doc in CLEAN_DOCS:
    if "tokens" in doc:
        vocab_set.update(doc["tokens"])
    if "title" in doc:
        title_words = doc["title"].lower().replace("-", " ").replace(":", " ").split()
        vocab_set.update(title_words)

print(f" Loaded {len(CLEAN_DOCS)} documents")
print(f" Vocabulary size: {len(vocab_set)}")

# ============================================================
#  PREPROCESSING UTILS
# ============================================================

def preprocess_query(q):
    q = q.lower().replace("-", " ")
    return q.split()

def compute_tf(tokens):
    vec = np.zeros(len(vocab_index))
    count = Counter(tokens)
    for word, freq in count.items():
        if word in vocab_index:
            vec[vocab_index[word]] = freq
    return vec

def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ============================================================
#  IMPROVED AUTOCORRECT
# ============================================================

def autocorrect_query(query):
    """Auto-correct typos in query using vocabulary"""
    words = query.lower().split()
    corrected_words = []
    corrections_made = []
    
    for word in words:
        # Skip very short words
        if len(word) <= 2:
            corrected_words.append(word)
            continue
            
        if word in vocab_set:
            corrected_words.append(word)
        else:
            # Find close matches with lower threshold
            matches = get_close_matches(word, vocab_set, n=1, cutoff=0.5)
            if matches:
                corrected_words.append(matches[0])
                corrections_made.append({"original": word, "corrected": matches[0]})
            else:
                corrected_words.append(word)
    
    corrected_query = " ".join(corrected_words)
    return corrected_query, corrections_made

# ============================================================
#  TITLE  DOCUMENT MAP
# ============================================================

title_map = { d["title"]: d for d in CLEAN_DOCS }

# ============================================================
#  JACCARD SIMILARITY
# ============================================================

def jaccard_similarity(q_tokens, doc_tokens, title_tokens):
    if len(q_tokens) == 0:
        return 0.0

    q_set = set(q_tokens)
    d_set = set(doc_tokens)

    inter = len(q_set & d_set)
    union = len(q_set | d_set)
    base = inter / union if union > 0 else 0.0

    # BOOST judul
    if len(q_set & set(title_tokens)) > 0:
        base += 0.3

    return base

# ============================================================
#  EVALUATION FUNCTIONS
# ============================================================

def precision_at_k(results, relevant, k):
    results = results[:k]
    hits = sum(1 for r in results if r in relevant)
    return hits / max(len(results), 1)

def recall_at_k(results, relevant, k):
    results = results[:k]
    hits = sum(1 for r in results if r in relevant)
    return hits / max(len(relevant), 1)

def f1_score(p, r):
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)

def accuracy_at_k(results, relevant, k):
    """Calculate accuracy as: (relevant retrieved) / total documents considered"""
    results = results[:k]
    hits = sum(1 for r in results if r in relevant)
    return hits / k if k > 0 else 0.0

def evaluate_search(query, results, method, runtime_ms):
    """Evaluate search performance based on score distribution"""
    if not results:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "accuracy": round(runtime_ms, 2),
            "query": query,
            "method": method
        }

    # Get scores from results (either tfidf_score or jaccard_score depending on method)
    scores = []
    for r in results[:10]:
        if method == "tfidf" and "tfidf_score" in r:
            scores.append(r["tfidf_score"])
        elif method == "jaccard" and "jaccard_score" in r:
            scores.append(r["jaccard_score"])
        elif "score" in r:  # For hybrid
            scores.append(r["score"])
    
    if not scores:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "accuracy": round(runtime_ms, 2),
            "query": query,
            "method": method
        }
    
    # Calculate metrics from scores
    # Precision: average of top 5 scores
    precision = sum(scores[:5]) / len(scores[:5]) if len(scores) >= 5 else sum(scores) / len(scores)
    
    # Recall: average of all 10 scores
    recall = sum(scores) / len(scores)
    
    # F1: harmonic mean
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Accuracy: score spread/differentiation (how well ranked results separate)
    # Based on: (max_score - min_score) / max_score
    # This measures how much the top result differs from worst result
    # Higher spread = better differentiation = higher accuracy
    if len(scores) > 0:
        max_score = max(scores)
        min_score = min(scores)
        
        if max_score > 0:
            # Score spread: how much the best differs from worst (as percentage of max)
            spread = (max_score - min_score) / max_score
            accuracy_pct = max(0, min(100, spread * 100))
        else:
            accuracy_pct = 0.0
    else:
        accuracy_pct = 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(runtime_ms, 2),
        "query": query,
        "method": method
    }# ============================================================
#  SEARCH FUNCTION
# ============================================================

def search(query, method="hybrid", top_k=10):
    import time
    start_time = time.time()
    
    # === AUTOCORRECT ===
    corrected_query, corrections = autocorrect_query(query)
    
    # === PREPROCESS ===
    q_tokens = preprocess_query(corrected_query)

    # ======================================================
    # TF-IDF SEARCH (WITH TITLE BOOST)
    # ======================================================

    q_tf = compute_tf(q_tokens)
    q_vec = q_tf * idf

    tfidf_scores = []

    for i, doc_vec in enumerate(tfidf_matrix):
        sim = cosine_similarity(q_vec, doc_vec)

        # TITLE BOOST TF-IDF
        title = filenames[i].lower().replace("-", " ")
        title_tokens = title.split()

        if len(set(q_tokens) & set(title_tokens)) > 0:
            sim += 0.3

        tfidf_scores.append((i, sim))

    tfidf_ranked = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    tfidf_ranked = [(i, s) for (i, s) in tfidf_ranked if s > 0]

    # ======================================================
    # JACCARD SEARCH
    # ======================================================

    jaccard_scores = []
    for doc in CLEAN_DOCS:
        sim = jaccard_similarity(
            q_tokens,
            doc["tokens"],
            doc["title"].lower().replace("-", " ").split()
        )
        jaccard_scores.append((doc["title"], sim))

    jaccard_ranked = sorted(jaccard_scores, key=lambda x: x[1], reverse=True)
    jaccard_ranked = [(t, s) for (t, s) in jaccard_ranked if s > 0]

    # ======================================================
    # BUILD RESULTS (with new field names)
    # ======================================================

    result_tfidf = []
    for idx, score in tfidf_ranked[:top_k]:
        title = filenames[idx]
        if title in title_map:
            d = title_map[title]
            result_tfidf.append({
                "title": d["title"],
                "poster": d["poster"],
                "description": d["description"],
                "tfidf_score": float(score)
            })

    result_jaccard = []
    for title, score in jaccard_ranked[:top_k]:
        if title in title_map:
            d = title_map[title]
            result_jaccard.append({
                "title": d["title"],
                "poster": d["poster"],
                "description": d["description"],
                "jaccard_score": float(score)
            })

    # ======================================================
    # HYBRID: COMBINE BOTH
    # ======================================================

    if method == "hybrid":
        combined = {}

        for item in result_tfidf:
            title = item["title"]
            combined[title] = {
                "data": item,
                "tfidf_score": item["tfidf_score"],
                "jaccard_score": 0
            }

        for item in result_jaccard:
            title = item["title"]
            if title in combined:
                combined[title]["jaccard_score"] = item["jaccard_score"]
            else:
                combined[title] = {
                    "data": item,
                    "tfidf_score": 0,
                    "jaccard_score": item["jaccard_score"]
                }

        hybrid_results = []
        for title, scores in combined.items():
            hybrid_score = (scores["tfidf_score"] * 0.7 + scores["jaccard_score"] * 0.3)
            result = scores["data"].copy()
            result["score"] = float(hybrid_score)
            result["tfidf_score"] = float(scores["tfidf_score"])
            result["jaccard_score"] = float(scores["jaccard_score"])
            hybrid_results.append(result)

        hybrid_results.sort(key=lambda x: x["score"], reverse=True)

        # Calculate runtime and evaluations for TF-IDF and Jaccard (no hybrid in evaluation)
        # Use corrected_query for evaluation to match the actual search results
        runtime_ms = (time.time() - start_time) * 1000
        eval_tfidf = evaluate_search(corrected_query, result_tfidf[:top_k], "tfidf", runtime_ms)
        eval_jaccard = evaluate_search(corrected_query, result_jaccard[:top_k], "jaccard", runtime_ms)

        evaluation = {
            "tfidf": eval_tfidf,
            "jaccard": eval_jaccard
        }

        return {
            "results": hybrid_results[:top_k],
            "corrected_query": corrected_query,
            "corrections": corrections,
            "total": len(hybrid_results),
            "evaluation": evaluation
        }

    elif method == "tfidf":
        # Calculate runtime and evaluations (provide both tfidf and jaccard for comparison)
        # Use corrected_query for evaluation to match the actual search results
        runtime_ms = (time.time() - start_time) * 1000
        eval_tfidf = evaluate_search(corrected_query, result_tfidf[:top_k], "tfidf", runtime_ms)
        eval_jaccard = evaluate_search(corrected_query, result_jaccard[:top_k], "jaccard", runtime_ms)

        evaluation = {
            "tfidf": eval_tfidf,
            "jaccard": eval_jaccard
        }

        return {
            "results": result_tfidf,
            "corrected_query": corrected_query,
            "corrections": corrections,
            "total": len(result_tfidf),
            "evaluation": evaluation
        }

    elif method == "jaccard":
        # Calculate runtime and evaluations (provide both tfidf and jaccard for comparison)
        # Use corrected_query for evaluation to match the actual search results
        runtime_ms = (time.time() - start_time) * 1000
        eval_tfidf = evaluate_search(corrected_query, result_tfidf[:top_k], "tfidf", runtime_ms)
        eval_jaccard = evaluate_search(corrected_query, result_jaccard[:top_k], "jaccard", runtime_ms)

        evaluation = {
            "tfidf": eval_tfidf,
            "jaccard": eval_jaccard
        }

        return {
            "results": result_jaccard,
            "corrected_query": corrected_query,
            "corrections": corrections,
            "total": len(result_jaccard),
            "evaluation": evaluation
        }

# ============================================================
#  API ENDPOINTS
# ============================================================

@app.route('/api/search', methods=['GET', 'POST'])
def api_search():
    """Search endpoint"""
    if request.method == 'POST':
        data = request.get_json()
        query = data.get('query', '')
        method = data.get('method', 'hybrid')
        top_k = data.get('top_k', 10)
    else:
        query = request.args.get('query', '')
        method = request.args.get('method', 'hybrid')
        top_k = int(request.args.get('top_k', 10))

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    try:
        result = search(query, method, top_k)

        return jsonify({
            'query': query,
            'corrected_query': result['corrected_query'] if result['corrected_query'] != query else None,
            'corrections': result['corrections'] if result['corrections'] else None,
            'method': method,
            'results': result['results'],
            'evaluation': result.get('evaluation')
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'total_documents': len(CLEAN_DOCS),
        'vocabulary_size': len(vocab_set)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  SciFind Search Engine API Server")
    print("="*60)
    print(f"  Backend ready with {len(CLEAN_DOCS)} sci-fi movies/series")
    print(f"  Server: http://localhost:5000")
    print(f"  Health: http://localhost:5000/api/health")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
