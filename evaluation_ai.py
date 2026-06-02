"""
evaluation_ai.py

Evaluasi komparatif tiga metode pencarian:
- TF-IDF + Cosine Similarity
- Jaccard Similarity
- Semantic Search (MiniLM)

Metrik yang digunakan:
- Precision@K
- Recall@K
- F1-Score
- Latency (ms)
"""

import time
import json
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

# ============================================================
#  LOAD DATA
# ============================================================

print("Memuat data...")

with open("clean_documents.json", "r", encoding="utf-8") as f:
    CLEAN_DOCS = json.load(f)

with open("tfidf_index.json", "r", encoding="utf-8") as f:
    TFIDF = json.load(f)

with open("embeddings.json", "r", encoding="utf-8") as f:
    raw_embeddings = json.load(f)

tfidf_matrix = np.array(TFIDF["matrix"])
vocab = TFIDF["vocab"]
vocab_index = {w: i for i, w in enumerate(vocab)}
idf = np.array(TFIDF["idf"])
filenames = TFIDF["filenames"]

doc_ids = list(raw_embeddings.keys())
embedding_matrix = np.array([raw_embeddings[d]["embedding"] for d in doc_ids])
norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
norms[norms == 0] = 1
embedding_matrix_norm = embedding_matrix / norms

print(f"Dokumen: {len(CLEAN_DOCS)} | Vocab: {len(vocab)} | Embeddings: {len(doc_ids)}")

# ============================================================
#  LOAD MODEL SEMANTIC
# ============================================================

print("Memuat model MiniLM...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Model siap.")

# ============================================================
#  PREPROCESSING
# ============================================================

def preprocess_query(q):
    return q.lower().replace("-", " ").split()

def compute_tf(tokens):
    vec = np.zeros(len(vocab_index))
    counter = Counter(tokens)
    for w, f in counter.items():
        if w in vocab_index:
            vec[vocab_index[w]] = f
    return vec

def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def jaccard_sim(q_tokens, doc_tokens, title_tokens):
    q = set(q_tokens)
    d = set(doc_tokens)
    if len(q) == 0:
        return 0.0
    inter = len(q & d)
    union = len(q | d)
    score = inter / union if union > 0 else 0.0
    if len(q & set(title_tokens)) > 0:
        score += 0.3
    return score

# ============================================================
#  SEARCH FUNCTIONS
# ============================================================

def search_tfidf(q_tokens, top_k=10):
    q_tf = compute_tf(q_tokens)
    q_vec = q_tf * idf

    scores = []
    for i, doc_vec in enumerate(tfidf_matrix):
        sim = cosine_sim(q_vec, doc_vec)
        title_tokens = filenames[i].lower().replace("-", " ").split()
        if len(set(q_tokens) & set(title_tokens)) > 0:
            sim += 0.3
        scores.append((filenames[i], sim))

    scores = [(t, s) for t, s in scores if s > 0]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scores[:top_k]]

def search_jaccard(q_tokens, top_k=10):
    scores = []
    for doc in CLEAN_DOCS:
        title_tokens = doc["title"].lower().replace("-", " ").split()
        sim = jaccard_sim(q_tokens, doc["tokens"], title_tokens)
        if sim > 0:
            scores.append((doc["title"], sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scores[:top_k]]

def search_semantic(query, top_k=10):
    q_emb = model.encode(query, convert_to_numpy=True)
    q_norm = np.linalg.norm(q_emb)
    if q_norm == 0:
        return []
    q_emb = q_emb / q_norm

    sims = embedding_matrix_norm @ q_emb
    top_indices = np.argsort(sims)[::-1][:top_k]

    # Buat lookup dari doc_id ke title langsung dari clean_documents
    doc_id_to_title = {}
    for doc in CLEAN_DOCS:
        title = doc["title"]
        # Simpan berbagai variasi key
        doc_id_to_title[title] = title
        doc_id_to_title[title.replace(" ", "_")] = title
        doc_id_to_title[title.lower()] = title
        doc_id_to_title[title.lower().replace(" ", "_")] = title

    results = []
    for idx in top_indices:
        doc_id = doc_ids[idx]
        # Coba berbagai variasi matching
        title = (
            doc_id_to_title.get(doc_id) or
            doc_id_to_title.get(doc_id.replace("_", " ")) or
            doc_id_to_title.get(doc_id.lower()) or
            doc_id_to_title.get(doc_id.lower().replace("_", " "))
        )
        if title:
            results.append(title)

    return results

# ============================================================
#  METRIK EVALUASI
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
    return 2 * p * r / (p + r)

# ============================================================
#  TEST SET
# ============================================================

TEST_SET = [
    ("interstellar", ["Interstellar"]),
    ("i am legend", ["I Am Legend"]),
    ("avengers", ["Avengers Endgame", "Avengers Age of Ultron", "The Avengers"]),
    ("spider", [
        "Spider-Man No Way Home", "Spider-Man Homecoming",
        "Spider-Man Far From Home", "Spider-Man Across the Spider-Verse"
    ]),
    ("the batman", ["The Batman", "The Batman Part II"]),
    ("perjalanan luar angkasa eksplorasi", ["Interstellar", "Gravity", "The Martian", "2001 A Space Odyssey"]),
    ("film tentang kecerdasan buatan yang mempertanyakan kemanusiaan", ["Ex Machina", "2001 A Space Odyssey"]),
    ("manusia bertahan hidup dari wabah virus zombie", ["28 Days Later", "I Am Legend", "28 Weeks Later"]),
    ("superhero melawan penjahat kota gotham", ["The Batman", "The Batman Part II", "Batman Begins"]),
]

K = 10

# ============================================================
#  EVALUASI
# ============================================================

def evaluate_all():
    all_results = []

    tfidf_p_list, tfidf_r_list, tfidf_f1_list, tfidf_lat_list = [], [], [], []
    jaccard_p_list, jaccard_r_list, jaccard_f1_list, jaccard_lat_list = [], [], [], []
    semantic_p_list, semantic_r_list, semantic_f1_list, semantic_lat_list = [], [], [], []

    print("\n" + "=" * 60)
    print("  EVALUASI KOMPARATIF: TF-IDF vs JACCARD vs SEMANTIC")
    print("=" * 60)

    for query, relevant in TEST_SET:
        q_tokens = preprocess_query(query)

        # TF-IDF
        t0 = time.time()
        tfidf_res = search_tfidf(q_tokens, K)
        tfidf_lat = (time.time() - t0) * 1000
        tp = precision_at_k(tfidf_res, relevant, K)
        tr = recall_at_k(tfidf_res, relevant, K)
        tf1 = f1_score(tp, tr)

        # Jaccard
        t0 = time.time()
        jaccard_res = search_jaccard(q_tokens, K)
        jaccard_lat = (time.time() - t0) * 1000
        jp = precision_at_k(jaccard_res, relevant, K)
        jr = recall_at_k(jaccard_res, relevant, K)
        jf1 = f1_score(jp, jr)

        # Semantic
        t0 = time.time()
        semantic_res = search_semantic(query, K)
        semantic_lat = (time.time() - t0) * 1000
        sp = precision_at_k(semantic_res, relevant, K)
        sr = recall_at_k(semantic_res, relevant, K)
        sf1 = f1_score(sp, sr)

        tfidf_p_list.append(tp);    tfidf_r_list.append(tr);    tfidf_f1_list.append(tf1);    tfidf_lat_list.append(tfidf_lat)
        jaccard_p_list.append(jp);  jaccard_r_list.append(jr);  jaccard_f1_list.append(jf1);  jaccard_lat_list.append(jaccard_lat)
        semantic_p_list.append(sp); semantic_r_list.append(sr); semantic_f1_list.append(sf1); semantic_lat_list.append(semantic_lat)

        print(f"\nQuery: \"{query}\"")
        print(f"Relevan : {relevant}")
        print(f"{'Metode':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Latency':>12}")
        print("-" * 54)
        print(f"{'TF-IDF':<12} {tp:>10.4f} {tr:>8.4f} {tf1:>8.4f} {tfidf_lat:>10.2f} ms")
        print(f"{'Jaccard':<12} {jp:>10.4f} {jr:>8.4f} {jf1:>8.4f} {jaccard_lat:>10.2f} ms")
        print(f"{'Semantic':<12} {sp:>10.4f} {sr:>8.4f} {sf1:>8.4f} {semantic_lat:>10.2f} ms")

        all_results.append({
            "query": query,
            "relevant": relevant,
            "tfidf":    {"precision": tp, "recall": tr, "f1": tf1, "latency_ms": round(tfidf_lat, 2)},
            "jaccard":  {"precision": jp, "recall": jr, "f1": jf1, "latency_ms": round(jaccard_lat, 2)},
            "semantic": {"precision": sp, "recall": sr, "f1": sf1, "latency_ms": round(semantic_lat, 2)},
        })

    # Rata-rata
    print("\n" + "=" * 60)
    print("  RINGKASAN (RATA-RATA SEMUA QUERY)")
    print("=" * 60)
    print(f"{'Metode':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Latency':>12}")
    print("-" * 54)
    print(f"{'TF-IDF':<12} {np.mean(tfidf_p_list):>10.4f} {np.mean(tfidf_r_list):>8.4f} {np.mean(tfidf_f1_list):>8.4f} {np.mean(tfidf_lat_list):>10.2f} ms")
    print(f"{'Jaccard':<12} {np.mean(jaccard_p_list):>10.4f} {np.mean(jaccard_r_list):>8.4f} {np.mean(jaccard_f1_list):>8.4f} {np.mean(jaccard_lat_list):>10.2f} ms")
    print(f"{'Semantic':<12} {np.mean(semantic_p_list):>10.4f} {np.mean(semantic_r_list):>8.4f} {np.mean(semantic_f1_list):>8.4f} {np.mean(semantic_lat_list):>10.2f} ms")
    print("=" * 60)

    summary = {
        "tfidf":    {"avg_precision": round(np.mean(tfidf_p_list), 4), "avg_recall": round(np.mean(tfidf_r_list), 4), "avg_f1": round(np.mean(tfidf_f1_list), 4), "avg_latency_ms": round(np.mean(tfidf_lat_list), 2)},
        "jaccard":  {"avg_precision": round(np.mean(jaccard_p_list), 4), "avg_recall": round(np.mean(jaccard_r_list), 4), "avg_f1": round(np.mean(jaccard_f1_list), 4), "avg_latency_ms": round(np.mean(jaccard_lat_list), 2)},
        "semantic": {"avg_precision": round(np.mean(semantic_p_list), 4), "avg_recall": round(np.mean(semantic_r_list), 4), "avg_f1": round(np.mean(semantic_f1_list), 4), "avg_latency_ms": round(np.mean(semantic_lat_list), 2)},
    }

    output = {"per_query": all_results, "summary": summary}
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nHasil disimpan ke: evaluation_results.json")

    return output

# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    evaluate_all()
