from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from typing import List

_MODEL = None

def get_text_embedder():
    global _MODEL
    if _MODEL is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    return _MODEL

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_text_embedder()
    batch_size = 32
    embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        arr = model.encode(chunk, show_progress_bar=False)
        embs.append(arr)
    return np.vstack(embs)
