"""
===========================================================
MEDQUAD RETRIEVER MODULE
Health Buddy – Generative AI Project

Purpose:
    - Load MedQuAD CSV
    - Build embeddings for (question + answer)
    - Retrieve top-k most relevant medical QA pairs
    - Used for grounding the Generative AI responses

Embeddings saved:
    stored_models/MedQuAD/index_<timestamp>/
        embeddings.npy
        metadata.json
===========================================================
"""

import os
import json
import numpy as np
import pandas as pd
import datetime
import torch
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# 1) PATHS
# =========================================================
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

csv_path = "datasets/MedQuAD/medquad.csv"

timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
SAVE_PATH = f"stored_models/MedQuAD/index_{timestamp}"
os.makedirs(SAVE_PATH, exist_ok=True)

print("📁 Saving retriever index to:", SAVE_PATH)

# =========================================================
# 2) LOAD DATASET
# =========================================================
df = pd.read_csv(csv_path)
df = df.dropna(subset=["question", "answer"])

texts = []
metadata = []

for _, row in df.iterrows():
    qa_text = f"Q: {row['question']} A: {row['answer']}"
    texts.append(qa_text)

    metadata.append({
        "question": row["question"],
        "answer": row["answer"],
        "source": row.get("source", ""),
        "focus_area": row.get("focus_area", "")
    })

print(f"📦 Loaded {len(texts)} MedQuAD entries.")

# =========================================================
# 3) LOAD EMBEDDING MODEL
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)
print("🔥 Using device for ST embeddings:", device)

# Optional: build Gemini embeddings as well if key present
gem_build = False
gem_model_name = "text-embedding-004"
try:
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        gem_build = os.getenv("USE_GEMINI_EMBEDDINGS", "0") == "1"
        if gem_build:
            print("✨ Building Gemini embeddings with:", gem_model_name)
    else:
        print("ℹ️ GEMINI_API_KEY not set; skipping Gemini embedding build")
except Exception:
    print("⚠️ google-generativeai not available; skipping Gemini embedding build")

# =========================================================
# 4) GENERATE EMBEDDINGS VECTOR
# =========================================================
print("⚙️ Generating embeddings...")

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    batch_size=32,
    show_progress_bar=True
)

if gem_build:
    gem_vecs = []
    for t in texts:
        try:
            resp = genai.embed_content(model=gem_model_name, content=t)
            gem_vecs.append(np.array(resp.get("embedding", []), dtype=np.float32))
        except Exception:
            gem_vecs.append(np.zeros(1, dtype=np.float32))
    max_len = max(v.size for v in gem_vecs) if gem_vecs else 0
    gem_aligned = []
    for v in gem_vecs:
        if v.size < max_len:
            pad = np.zeros(max_len - v.size, dtype=np.float32)
            gem_aligned.append(np.concatenate([v, pad]))
        else:
            gem_aligned.append(v)
    gem_arr = np.stack(gem_aligned) if gem_aligned else np.zeros((0,0), dtype=np.float32)
    np.save(os.path.join(SAVE_PATH, "embeddings.npy"), gem_arr)
else:
    np.save(os.path.join(SAVE_PATH, "embeddings.npy"), embeddings)

with open(os.path.join(SAVE_PATH, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

with open(os.path.join(SAVE_PATH, "index_meta.json"), "w") as f:
    json.dump({
        "st_model": MODEL_NAME,
        "gemini_model": gem_model_name if gem_build else None,
        "built_at": timestamp
    }, f, indent=2)

print("✅ Embeddings saved!")

# ---------------------------------------------------------
# Load embeddings as *torch tensor on same device*
# ---------------------------------------------------------
embeddings_tensor = torch.from_numpy(embeddings).to(device)

# =========================================================
# 5) RETRIEVAL FUNCTION
# =========================================================
def retrieve_medquad(query: str, top_k: int = 5):
    """Retrieve top-k most relevant QA pairs from MedQuAD."""
    
    # Encode query → tensor ON SAME DEVICE
    q_emb = model.encode(query, convert_to_tensor=True).to(device)

    # Cosine similarity ON SAME DEVICE
    scores = util.cos_sim(q_emb, embeddings_tensor)[0]

    top_scores, top_indices = torch.topk(scores, k=top_k)

    results = []
    for score, idx in zip(top_scores, top_indices):
        idx = int(idx)
        results.append({
            "score": float(score),
            "question": metadata[idx]["question"],
            "answer": metadata[idx]["answer"],
            "source": metadata[idx]["source"],
            "focus_area": metadata[idx]["focus_area"]
        })

    return results

# =========================================================
# 6) DEMO QUERY
# =========================================================
if __name__ == "__main__":
    print("\n🔍 DEMO: Searching MedQuAD")
    q = "What causes glaucoma?"
    results = retrieve_medquad(q, top_k=3)

    for r in results:
        print("\n---")
        print("Score:", r["score"])
        print("Q:", r["question"])
        print("A:", r["answer"][:200], "...")
