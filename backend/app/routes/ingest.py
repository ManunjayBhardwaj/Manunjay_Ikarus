from fastapi import APIRouter
import os

router = APIRouter()


@router.post("/api/ingest")
def ingest():
    # default to the cleaned dataset produced by the cleaning script
    import json
    import pandas as pd
    from ..utils.data import load_and_clean

    default_path = os.path.join(os.getcwd(), "backend", "data", "cleaned_furniture.csv")
    path = os.getenv('DATASET_PATH', default_path)

    if not os.path.exists(path):
        return {"status": "no dataset found", "path": path}

    # load and canonicalize
    df = load_and_clean(path)

    # try to create embeddings and push to vector DB; if heavy ML libs or DB creds
    # are missing, fall back to writing metadata only so the UI still works.
    try:
        # lazy imports that may fail in minimal environments
        from ..models.embeddings import embed_texts
        from ..models.nlp_utils import fit_product_clusters
        from ..db.vector_db import upsert_embeddings

        texts = (df['title'].fillna('') + ' ' + df['description'].fillna('')).tolist()
        embs = embed_texts(texts)
        labels, stats = fit_product_clusters(embs, n_clusters=12)
        df['cluster_label'] = labels

        items = []
        for i, row in df.iterrows():
            meta = row.to_dict()
            # ensure any numpy arrays are converted
            try:
                meta['embedding'] = embs[i].tolist()
            except Exception:
                meta['embedding'] = None
            items.append({'id': str(row.get('uniq_id', i)), 'embedding': embs[i], 'metadata': meta})

        upsert_embeddings(items)

        # also update local metadata store (if vector DB upsert doesn't overwrite it)
        storage_dir = os.path.join(os.getcwd(), "backend", "backend", "storage")
        os.makedirs(storage_dir, exist_ok=True)
        meta_path = os.path.join(storage_dir, 'metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            # write a dict id->metadata
            j = {str(row.get('uniq_id', i)): row.to_dict() for i, row in df.iterrows()}
            json.dump(j, f, ensure_ascii=False, indent=2)

        return {"status": "ingested", "count": len(items), "cluster_stats": stats}

    except Exception as exc:  # pragma: no cover - fallback path
        # save metadata JSON so UI and sample endpoints can use cleaned data
        storage_dir = os.path.join(os.getcwd(), "backend", "backend", "storage")
        os.makedirs(storage_dir, exist_ok=True)
        meta_path = os.path.join(storage_dir, 'metadata.json')
        out = {str(row.get('uniq_id', i)): row.to_dict() for i, row in df.iterrows()}
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        return {"status": "ingested_no_embeddings", "count": len(df), "reason": str(exc)}
