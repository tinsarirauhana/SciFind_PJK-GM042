"""
generate_embeddings.py

Jalankan file ini SEKALI untuk menghasilkan embeddings dari seluruh dokumen.
Output: embeddings.json (disimpan di folder yang sama)

Cara pakai:
    python generate_embeddings.py

Requirement tambahan:
    pip install sentence-transformers
"""

import os
import json
import time
from sentence_transformers import SentenceTransformer

# ============================================================
#  KONFIGURASI
# ============================================================

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DATASET_DIR = "dataset_clean"
OUTPUT_FILE = "embeddings.json"

# ============================================================
#  LOAD MODEL
# ============================================================

print(f"Memuat model {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("Model berhasil dimuat.")

# ============================================================
#  LOAD DOKUMEN
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(SCRIPT_DIR, DATASET_DIR)

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_path}")

files = sorted([f for f in os.listdir(dataset_path) if f.endswith(".json")])
print(f"Ditemukan {len(files)} dokumen di {DATASET_DIR}/")

# ============================================================
#  GENERATE EMBEDDINGS
# ============================================================

embeddings_data = {}
total = len(files)
start_time = time.time()

for i, filename in enumerate(files):
    filepath = os.path.join(dataset_path, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        doc = json.load(f)

    tokens = doc.get("tokens", [])
    text = " ".join(tokens)

    embedding = model.encode(text, convert_to_numpy=True)

    doc_id = filename.replace(".json", "")
    embeddings_data[doc_id] = {
        "embedding": embedding.tolist()
    }

    if (i + 1) % 50 == 0 or (i + 1) == total:
        elapsed = time.time() - start_time
        print(f"  [{i+1}/{total}] selesai | waktu: {elapsed:.1f}s")

# ============================================================
#  SIMPAN OUTPUT
# ============================================================

output_path = os.path.join(SCRIPT_DIR, OUTPUT_FILE)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(embeddings_data, f)

elapsed_total = time.time() - start_time
print(f"\nSelesai. {total} embeddings disimpan ke: {output_path}")
print(f"Total waktu: {elapsed_total:.1f} detik")
