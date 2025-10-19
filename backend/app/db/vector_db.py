import os
from typing import List, Dict
import json
import numpy as np

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss")
INDEX_NAME = os.getenv("PINECONE_INDEX", "furniture-index")

if VECTOR_BACKEND == "pinecone":
    try:
        import pinecone
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
    except Exception:
        pinecone = None

class VectorDB:
    def __init__(self):
        self.backend = VECTOR_BACKEND
        self.index = None
        self.metadata = {}
        self._init_index(INDEX_NAME)

    def _init_index(self, index_name):
        if self.backend == "pinecone" and 'pinecone' in globals() and pinecone is not None:
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(index_name, dimension=384)
            self.index = pinecone.Index(index_name)
        else:
            # FAISS fallback
            import faiss
            self.faiss = faiss
            self.index_path = os.path.join(os.getcwd(), "backend", "storage", "faiss_index.bin")
            self.meta_path = os.path.join(os.getcwd(), "backend", "storage", "metadata.json")
            self._load_faiss()

    def _load_faiss(self):
        d = 384
        if os.path.exists(self.index_path):
            self.index = self.faiss.read_index(self.index_path)
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r") as f:
                    self.metadata = json.load(f)
        else:
            self.index = self.faiss.IndexFlatIP(d)
            self.metadata = {}

    def init_index(self, index_name):
        return self._init_index(index_name)

    def upsert_embeddings(self, items: List[Dict]):
        # items: list of {'id','embedding','metadata'}
        if self.backend == "pinecone" and 'pinecone' in globals() and pinecone is not None:
            to_upsert = [(it['id'], it['embedding'].tolist(), it.get('metadata', {})) for it in items]
            self.index.upsert(vectors=to_upsert)
        else:
            vecs = np.vstack([it['embedding'] for it in items]).astype('float32')
            # normalize for IP
            self.faiss.normalize_L2(vecs)
            self.index.add(vecs)
            # store metadata
            start = len(self.metadata)
            for i, it in enumerate(items):
                self.metadata[str(start + i)] = it.get('metadata', {})
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            self.faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "w") as f:
                json.dump(self.metadata, f)

    def query(self, text_embedding, top_k=5):
        if self.backend == "pinecone" and 'pinecone' in globals() and pinecone is not None:
            res = self.index.query(vector=text_embedding.tolist(), top_k=top_k, include_metadata=True)
            matches = []
            for m in res['matches']:
                matches.append({
                    'id': m['id'],
                    'score': m['score'],
                    'metadata': m.get('metadata', {})
                })
            return matches
        else:
            if len(text_embedding.shape) == 1:
                q = text_embedding.reshape(1, -1).astype('float32')
            else:
                q = text_embedding.astype('float32')
            self.faiss.normalize_L2(q)
            D, I = self.index.search(q, top_k)
            results = []
            for score_row, idx_row in zip(D, I):
                for sc, idx in zip(score_row, idx_row):
                    meta = self.metadata.get(str(idx), {})
                    results.append({
                        'id': str(idx),
                        'score': float(sc),
                        'metadata': meta
                    })
            return results


_DB = VectorDB()

def init_index(index_name):
    return _DB.init_index(index_name)

def upsert_embeddings(items: List[Dict]):
    return _DB.upsert_embeddings(items)

def query(text_embedding, top_k=5):
    return _DB.query(text_embedding, top_k=top_k)
