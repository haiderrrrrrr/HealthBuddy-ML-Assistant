import os
import json
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def _resolve_latest_index_dir():
    root = os.path.join(BASE_DIR, "stored_models", "MedQuAD", "latest")
    subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    subdirs.sort()
    if not subdirs:
        raise FileNotFoundError(f"No index found under {root}")
    return os.path.join(root, subdirs[-1])

def _ensure_encoder():
    try:
        from sentence_transformers import SentenceTransformer, util
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        return model, util, device
    except Exception:
        return None, None, None

def _ensure_gemini():
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None, None
        genai.configure(api_key=api_key)
        model_name = "text-embedding-004"
        return genai, model_name
    except Exception:
        return None, None

def _tfidf_fallback(query: str, meta: list, k: int):
    docs = [f"{m.get('question','')} {m.get('answer','')}" for m in meta]
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(docs + [query])
    qv = X[-1]
    sims = (X[:-1] @ qv.T).toarray().squeeze(1)
    idxs = np.argsort(sims)[-min(k, len(meta)):][::-1]
    results = []
    for i in idxs:
        item = meta[int(i)]
        results.append({
            "score": float(sims[int(i)]),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "source": item.get("source", ""),
            "focus_area": item.get("focus_area", "")
        })
    return results

def _token_overlap(a: str, b: str) -> float:
    ta = {t for t in ''.join([c.lower() if c.isalnum() else ' ' for c in a]).split() if t}
    tb = {t for t in ''.join([c.lower() if c.isalnum() else ' ' for c in b]).split() if t}
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    return len(inter) / max(1, len(ta))

def retrieve_medquad(query: str, k: int = 3):
    index_dir = _resolve_latest_index_dir()
    with open(os.path.join(index_dir, "metadata.json"), "r") as f:
        meta = json.load(f)
    idx_meta_path = os.path.join(index_dir, "index_meta.json")
    gem_use = False
    gem_model = None
    if os.path.exists(idx_meta_path):
        try:
            m = json.load(open(idx_meta_path, "r"))
            gem_model = m.get("gemini_model")
            gem_use = bool(gem_model)
        except Exception:
            gem_use = False

    genai, gem_model_conf = _ensure_gemini()
    if gem_use and genai is not None:
        try:
            emb = np.load(os.path.join(index_dir, "embeddings.npy"))
            emb_t = torch.tensor(emb, dtype=torch.float32)
            resp = genai.embed_content(model=gem_model or gem_model_conf, content=query)
            q_vec = np.array(resp.get("embedding", []), dtype=np.float32)
            if q_vec.size == 0:
                raise RuntimeError("Empty Gemini embedding")
            q = torch.tensor(q_vec).unsqueeze(0)
            scores = torch.nn.functional.cosine_similarity(q, emb_t)
            pool_k = min(max(10, k * 3), len(meta))
            top_scores, top_idx = torch.topk(scores, k=pool_k)
            pool = []
            for s, idx in zip(top_scores, top_idx):
                item = meta[int(idx)]
                qtext = item.get("question", "")
                overlap = _token_overlap(query, qtext)
                final = 0.7 * float(s) + 0.3 * overlap
                pool.append((final, {
                    "score": float(s),
                    "question": qtext,
                    "answer": item.get("answer", ""),
                    "source": item.get("source", ""),
                    "focus_area": item.get("focus_area", "")
                }))
            pool.sort(key=lambda x: x[0], reverse=True)
            return [p[1] for p in pool[:k]]
        except Exception:
            pass

    # Fallback to sentence-transformers index
    enc, util, device = _ensure_encoder()
    if enc is None:
        return _tfidf_fallback(query, meta, k)
    try:
        emb = np.load(os.path.join(index_dir, "embeddings.npy"))
        emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
        q = enc.encode(query, convert_to_tensor=True, device=device).to(device)
        scores = util.cos_sim(q, emb_t)[0]
        pool_k = min(max(10, k * 3), len(meta))
        top_scores, top_idx = torch.topk(scores, k=pool_k)
        pool = []
        for s, idx in zip(top_scores, top_idx):
            item = meta[int(idx)]
            qtext = item.get("question", "")
            overlap = _token_overlap(query, qtext)
            final = 0.7 * float(s) + 0.3 * overlap
            pool.append((final, {
                "score": float(s),
                "question": qtext,
                "answer": item.get("answer", ""),
                "source": item.get("source", ""),
                "focus_area": item.get("focus_area", "")
            }))
        pool.sort(key=lambda x: x[0], reverse=True)
        return [p[1] for p in pool[:k]]
    except Exception:
        return _tfidf_fallback(query, meta, k)
