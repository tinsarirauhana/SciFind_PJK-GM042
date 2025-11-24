from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from difflib import get_close_matches
from collections import Counter
import os

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DOCS = []
TFIDF = {}
tfidf_matrix = np.array([])
vocab = []
vocab_index = {}
idf = np.array([])
filenames = []
vocab_set = set()

# Load data if files exist
try:
    clean_docs_path = os.path.join(SCRIPT_DIR, "../clean_documents.json")
    tfidf_index_path = os.path.join(SCRIPT_DIR, "../tfidf_index.json")
    with open(clean_docs_path, "r", encoding='utf-8') as f:
        CLEAN_DOCS = json.load(f)
    with open(tfidf_index_path, "r", encoding='utf-8') as f:
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
            title_words = doc["title"].lower().replace("-", " ").replace(":", " ").split()
            vocab_set.update(title_words)
except Exception as e:
    print(f"Error loading data: {e}")

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

def autocorrect_query(query):
    words = query.lower().split()
    corrected_words = []
    corrections_made = []
    for word in words:
        if len(word) <= 2:
            corrected_words.append(word)
            continue
        if word in vocab_set:
            corrected_words.append(word)
        else:
            matches = get_close_matches(word, vocab_set, n=1, cutoff=0.5)
            if matches:
                corrected_words.append(matches[0])
                corrections_made.append({"original": word, "corrected": matches[0]})
            else:
                corrected_words.append(word)
    corrected_query = " ".join(corrected_words)
    return corrected_query, corrections_made

title_map = { d["title"]: d for d in CLEAN_DOCS }

def jaccard_similarity(q_tokens, doc_tokens, title_tokens):
    if len(q_tokens) == 0:
        return 0.0
    q_set = set(q_tokens)
    d_set = set(doc_tokens)
    inter = len(q_set & d_set)
    union = len(q_set | d_set)
    base = inter / union if union > 0 else 0.0
    if len(q_set & set(title_tokens)) > 0:
        base += 0.3
    return base

def search(query, method="hybrid", top_k=10):
    corrected_query, corrections = autocorrect_query(query)
    q_tokens = preprocess_query(corrected_query)
    q_tf = compute_tf(q_tokens)
    q_vec = q_tf * idf
    tfidf_scores = []
    for i, doc_vec in enumerate(tfidf_matrix):
        sim = cosine_similarity(q_vec, doc_vec)
        title = filenames[i].lower().replace("-", " ")
        title_tokens = title.split()
        if len(set(q_tokens) & set(title_tokens)) > 0:
            sim += 0.3
        tfidf_scores.append((i, sim))
    tfidf_ranked = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    tfidf_ranked = [(i, s) for (i, s) in tfidf_ranked if s > 0]
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
        return {
            "results": hybrid_results[:top_k],
            "corrected_query": corrected_query,
            "corrections": corrections,
            "total": len(hybrid_results)
        }
    elif method == "tfidf":
        return {
            "results": result_tfidf,
            "corrected_query": corrected_query,
            "corrections": corrections,
            "total": len(result_tfidf)
        }
    elif method == "jaccard":
        return {
            "results": result_jaccard,
            "corrected_query": corrected_query,
            "corrections": corrections,
            "total": len(result_jaccard)
        }

@app.route('/api/search', methods=['GET', 'POST'])
def api_search():
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
            'results': result['results']
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'total_documents': len(CLEAN_DOCS),
        'vocabulary_size': len(vocab_set)
    })

# Vercel will use 'app' as entry point
handler = app
