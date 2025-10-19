import os
import json
import numpy as np
from typing import List



def _cos_sim(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))

def recommend_by_query(query: str, k: int = 5):
    # Try normal flow (use embedder + vector DB). If the embedding/model
    # stack or vector DB isn't available (e.g. sentence-transformers not
    # installed), fall back to returning the first k rows from the local
    # ingestion metadata so the frontend still shows real data and images.
    try:
        # import heavy deps lazily so module import doesn't fail when libs are missing
        from .embeddings import embed_texts, get_text_embedder
        from ..db.vector_db import query as query_db
        model = get_text_embedder()
        q_emb = model.encode([query])[0]
        hits = query_db(q_emb, top_k=k)
        # re-rank by cosine similarity
        out = []
        for h in hits:
            meta = h.get('metadata', {})
            # some vector DBs or older metadata files may store metadata as a list
            # (e.g. [field1, field2]) — coerce to a dict when possible
            if isinstance(meta, list) and len(meta) > 0 and isinstance(meta[0], dict):
                meta = meta[0]
            if not isinstance(meta, dict):
                # last resort: wrap into a dict with a single key
                meta = {"title": str(meta)}
            emb = np.array(meta.get('embedding', q_emb.tolist()))
            score = _cos_sim(q_emb, emb)
            out.append({
                'uniq_id': meta.get('uniq_id', h.get('id')),
                'title': meta.get('title', 'Unknown'),
                'brand': meta.get('brand', 'Unknown'),
                'price': meta.get('price', None),
                'image_url': meta.get('image_url', None) or meta.get('image') or (meta.get('images')[0] if isinstance(meta.get('images'), list) and meta.get('images') else None),
                'categories': meta.get('categories', None),
                'score': score,
                'metadata': meta
            })
        out = sorted(out, key=lambda x: x['score'], reverse=True)
        return out[:k]

    except Exception:
        # Fallback: try local embeddings.npy + metadata.json search; if not available,
        # return the first k metadata items so the UI shows images and info.
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        storage_dir = os.path.join(base_dir, 'backend', 'storage')
        meta_path = os.path.join(storage_dir, 'metadata.json')
        emb_path = os.path.join(storage_dir, 'embeddings.npy')
        meta_list = []
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                if isinstance(meta, dict):
                    meta_list = list(meta.values())
                elif isinstance(meta, list):
                    meta_list = meta
            except Exception:
                meta_list = []

        # Try numpy-based search if embeddings available and sentence-transformers installed
        try:
            if os.path.exists(emb_path) and meta_list:
                embs = np.load(emb_path)
                # ensure shapes align
                if embs.shape[0] == len(meta_list):
                    try:
                        from sentence_transformers import SentenceTransformer
                        embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                        q_emb = embedder.encode([query], convert_to_numpy=True)[0].astype('float32')
                        # normalize
                        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)
                        embs_unit = embs.astype('float32')
                        norms = np.linalg.norm(embs_unit, axis=1, keepdims=True) + 1e-12
                        embs_unit = embs_unit / norms
                        scores = (embs_unit @ q_norm).astype(float)
                        idxs = list(np.argsort(scores)[-k:][::-1])
                        out = []
                        for i in idxs:
                            m = meta_list[i]
                            img = m.get('image') or (m.get('images')[0] if isinstance(m.get('images'), list) and m.get('images') else None) or m.get('image_url') or m.get('primary_image')
                            out.append({
                                'uniq_id': m.get('uniq_id') or m.get('id'),
                                'title': m.get('title', m.get('title_final', 'Unknown')),
                                'brand': m.get('brand', m.get('brand_final', 'Unknown')),
                                'price': m.get('price') or m.get('price_final'),
                                'image_url': img,
                                'categories': m.get('categories', m.get('categories_list')),
                                'score': float(scores[i]),
                                'metadata': m
                            })
                        return out
                    except Exception:
                        # embedder missing or failed; fall back to simple head
                        pass
        except Exception:
            pass

        # Final fallback: first k items
        out = []
        for m in meta_list[:k]:
            img = m.get('image') or (m.get('images')[0] if isinstance(m.get('images'), list) and m.get('images') else None) or m.get('image_url') or m.get('primary_image')
            out.append({
                'uniq_id': m.get('uniq_id') or m.get('id'),
                'title': m.get('title', m.get('title_final', 'Unknown')),
                'brand': m.get('brand', m.get('brand_final', 'Unknown')),
                'price': m.get('price') or m.get('price_final'),
                'image_url': img,
                'categories': m.get('categories', m.get('categories_list')),
                'score': 1.0,
                'metadata': m
            })
        return out

def recommend_similar(uniq_id: str, k: int = 5):
    # find by scanning metadata (FAISS stores metadata separately)
    # This is a simple placeholder: in real setup, store embeddings mapped to uniq_id
    # Here we attempt to find the embedding in metadata
    # NOTE: vector_db.query expects an embedding; for simplicity, return empty list if not found
    return []

def query_db(embedding, top_k=5):
    # lazy import
    from ..db.vector_db import query
    return query(embedding, top_k=top_k)
