#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train text-embedding based recommendation + clustering on the cleaned dataset.

This is a more robust version of the training script that:
 - accommodates different cleaned CSV column names (falls back to raw columns)
 - supports a --dry-run that avoids importing heavy ML libs
 - computes silhouette using cosine distance on normalized vectors
 - can call the project's GenAI wrapper to generate creative descriptions (guarded)

Usage:
  python backend/scripts/train_models.py --data backend/data/cleaned_furniture.csv --out backend/backend/storage --clusters 12
  python backend/scripts/train_models.py --data backend/data/cleaned_furniture.csv --dry-run
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict

import numpy as np
import pandas as pd


def log(msg: str):
    print(f"[train] {msg}", flush=True)


def read_cleaned_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset: {path}")
    df = pd.read_csv(path)

    # map expected names to available columns
    col_map = {}
    # title
    if 'title_norm' in df.columns:
        col_map['title'] = 'title_norm'
    elif 'title' in df.columns:
        col_map['title'] = 'title'
    else:
        col_map['title'] = None

    # brand
    if 'brand_norm' in df.columns:
        col_map['brand'] = 'brand_norm'
    elif 'brand' in df.columns:
        col_map['brand'] = 'brand'
    else:
        col_map['brand'] = None

    # description
    if 'description_norm' in df.columns:
        col_map['description'] = 'description_norm'
    elif 'description' in df.columns:
        col_map['description'] = 'description'
    else:
        col_map['description'] = None

    # price
    if 'price_num' in df.columns:
        col_map['price'] = 'price_num'
    elif 'price' in df.columns:
        col_map['price'] = 'price'
    else:
        col_map['price'] = None

    # categories
    if 'categories_norm' in df.columns:
        col_map['categories'] = 'categories_norm'
    elif 'categories' in df.columns:
        col_map['categories'] = 'categories'
    else:
        col_map['categories'] = None

    # primary image
    if 'primary_image' in df.columns:
        col_map['image'] = 'primary_image'
    elif 'image' in df.columns:
        col_map['image'] = 'image'
    elif 'images' in df.columns:
        col_map['image'] = 'images'
    else:
        col_map['image'] = None

    # material / color
    col_map['material'] = 'material_std' if 'material_std' in df.columns else ('material' if 'material' in df.columns else None)
    col_map['color'] = 'color_std' if 'color_std' in df.columns else ('color' if 'color' in df.columns else None)

    # uniq_id
    if 'uniq_id' not in df.columns and 'id' in df.columns:
        df['uniq_id'] = df['id'].astype(str)
    if 'uniq_id' not in df.columns:
        # create stable uniq ids
        df['uniq_id'] = [f"r{i}" for i in range(len(df))]

    # normalize minimal fields
    df[col_map['title']] = df.get(col_map['title'], pd.Series(['Unknown Title'] * len(df))).fillna('Unknown Title').astype(str)
    if col_map['brand']:
        df[col_map['brand']] = df[col_map['brand']].fillna('Unknown').astype(str)
    if col_map['description']:
        df[col_map['description']] = df[col_map['description']].fillna('No description available.').astype(str)

    # categories might be list-like string
    def parse_cats(x):
        if pd.isna(x):
            return []
        s = str(x).strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                import ast
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return [str(t) for t in v]
            except Exception:
                pass
        return [t.strip() for t in s.split(',') if t.strip()]

    if col_map['categories']:
        df['categories_list'] = df[col_map['categories']].apply(parse_cats)
    else:
        df['categories_list'] = [[] for _ in range(len(df))]

    # extract primary image if images list exists
    def pick_image(x):
        if pd.isna(x):
            return None
        if isinstance(x, list) and x:
            return x[0]
        s = str(x)
        if s.startswith('[') and s.endswith(']'):
            try:
                import ast
                v = ast.literal_eval(s)
                if isinstance(v, list) and v:
                    return str(v[0])
            except Exception:
                pass
        # fallback: if comma-separated
        parts = [p.strip() for p in s.split(',') if p.strip()]
        return parts[0] if parts else None

    if col_map['image']:
        df['primary_image'] = df[col_map['image']].apply(pick_image)
    else:
        df['primary_image'] = [None] * len(df)

    # final canonical columns available to the rest of the script
    df['title_final'] = df.get(col_map['title'], df['title'])
    df['brand_final'] = df.get(col_map['brand'], df.get('brand', ''))
    df['desc_final'] = df.get(col_map['description'], df.get('description', ''))
    df['price_final'] = df.get(col_map['price'], df.get('price', None))
    df['material_final'] = df.get(col_map['material'], df.get('material', ''))
    df['color_final'] = df.get(col_map['color'], df.get('color', ''))

    return df


def build_corpus_row(r: pd.Series) -> str:
    cats = " > ".join(r.get("categories_list", [])[:3])
    parts = [
        r.get('title_final', ''),
        f"Brand: {r.get('brand_final', '')}",
        f"Material: {r.get('material_final', '')}",
        f"Color: {r.get('color_final', '')}",
        f"Categories: {cats}" if cats else '',
        r.get('desc_final', ''),
    ]
    return " | ".join([p for p in parts if p])


def embed_texts(model, texts: List[str], batch_size: int = 128) -> np.ndarray:
    # model is expected to expose .encode(..., convert_to_numpy=True)
    embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        embs.append(model.encode(chunk, show_progress_bar=False, convert_to_numpy=True))
    X = np.vstack(embs).astype('float32')
    return X


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def safe_json_dump(obj, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='backend/backend/storage')
    ap.add_argument('--clusters', type=int, default=12)
    ap.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    ap.add_argument('--dry-run', action='store_true', help='Validate data and build corpus without heavy imports')
    ap.add_argument('--genai', action='store_true', help='Generate creative descriptions using project GenAI wrapper (GEMINI_API_KEY required)')
    ap.add_argument('--genai-sample', type=int, default=10, help='How many items to generate copy for')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    log(f"Loading cleaned dataset: {args.data}")
    df = read_cleaned_csv(args.data)
    n = len(df)
    log(f"Rows loaded: {n}")

    log('Composing corpus')
    corpus = [build_corpus_row(r) for _, r in df.iterrows()]

    if args.dry_run:
        log('Dry-run complete. Corpus composed. Sample:')
        for i, c in enumerate(corpus[:3]):
            log(f'  {i}: {c[:200]}')
        return

    # now import heavy libs
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        log('ERROR: sentence-transformers not installed. Install with: pip install sentence-transformers')
        raise

    # Try to import faiss; if it fails, continue without FAISS (we'll still save embeddings/metadata)
    faiss = None
    try:
        import faiss
    except Exception as exc:
        log(f'WARNING: faiss import failed: {exc}. Falling back: embeddings and metadata will be saved but FAISS index will be skipped.')
        faiss = None

    # load embedder
    log(f'Loading embedder: {args.model}')
    emb_model = SentenceTransformer(args.model)

    # compute embeddings
    log('Embedding texts...')
    X = embed_texts(emb_model, corpus, batch_size=128)
    d = X.shape[1]
    log(f'Embeddings shape: {X.shape}')

    # normalize for cosine
    X_unit = l2_normalize(X)

    # clustering
    k = min(args.clusters, len(df)) if len(df) > 1 else 1
    if k < 2:
        log('Not enough rows for clustering; skipping KMeans.')
        labels = np.zeros((len(df),), dtype=int)
        sil = None
    else:
        log(f'Training KMeans with k={k}...')
        # sklearn expects n_init an int in older versions; use 10 for compatibility
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except Exception:
            log('ERROR: scikit-learn missing. Install scikit-learn')
            raise

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_unit)  # cluster on normalized vectors

        # silhouette with cosine: pass metric='cosine' and use the normalized vectors
        try:
            sil = float(silhouette_score(X_unit, labels, metric='cosine'))
        except Exception:
            sil = None
        log(f'KMeans trained. Silhouette: {sil}')

    index = None
    if faiss is not None:
        try:
            # build faiss index
            log('Building FAISS cosine index...')
            # faiss expects a contiguous numpy.float32 array
            X_unit_c = np.ascontiguousarray(X_unit.astype(np.float32))
            # force requirements so faiss can get the buffer pointer
            X_unit_c = np.require(X_unit_c, dtype=np.float32, requirements=['C', 'A'])
            d = X_unit_c.shape[1]
            index = faiss.IndexFlatIP(int(d))
            # pass a plain numpy array (contiguous, float32)
            index.add(X_unit_c)
        except Exception as exc:
            log(f'WARNING: faiss index build failed: {exc}. Continuing without FAISS index.')
            index = None
    else:
        log('Skipping FAISS index build because faiss is not available.')

    # save artifacts
    emb_path = os.path.join(args.out, 'embeddings.npy')
    faiss_path = os.path.join(args.out, 'faiss_index.bin')
    meta_path = os.path.join(args.out, 'metadata.json')
    clusters_path = os.path.join(args.out, 'cluster_labels.csv')
    report_path = os.path.join(args.out, 'training_report.json')

    log(f'Saving embeddings -> {emb_path}')
    np.save(emb_path, X.astype('float32'))

    if index is not None and faiss is not None:
        try:
            log(f'Saving FAISS index -> {faiss_path}')
            faiss.write_index(index, faiss_path)
        except Exception as exc:
            log(f'Failed to write FAISS index: {exc}')
    else:
        log('No FAISS index to save.')

    # build metadata
    log('Saving metadata...')
    meta: List[Dict] = []
    for i, r in df.reset_index(drop=True).iterrows():
        meta.append({
            'row': int(i),
            'uniq_id': str(r.get('uniq_id', '')),
            'title': str(r.get('title_final', '')),
            'brand': str(r.get('brand_final', '')),
            'price': None if pd.isna(r.get('price_final')) else float(r.get('price_final')),
            'categories': r.get('categories_list', []),
            'image_url': None if pd.isna(r.get('primary_image')) else str(r.get('primary_image')),
            'material': str(r.get('material_final', '')),
            'color': str(r.get('color_final', '')),
        })

    # optionally generate creative descriptions using project's genai wrapper
    if args.genai:
        try:
            # ensure backend path is importable
            backend_dir = os.path.join(os.getcwd(), 'backend')
            if backend_dir not in sys.path:
                sys.path.insert(0, backend_dir)
            from app.models import genai as genai_mod
        except Exception as exc:
            log(f'GenAI wrapper import failed: {exc}. Skipping generated copy.')
            genai_mod = None

        if genai_mod is not None:
            log(f'Generating creative copy for up to {args.genai_sample} items (this uses your GEMINI_API_KEY)')
            for i, item in enumerate(meta[: args.genai_sample]):
                try:
                    txt = genai_mod.generate_marketing_copy(item, temperature=0.2)
                    item['generated_copy'] = txt
                except Exception as exc:
                    item['generated_copy'] = None
                    log(f'  genai failed for {item.get("uniq_id")} -> {exc}')

    safe_json_dump(meta, meta_path)

    # save clusters
    log(f'Saving cluster labels -> {clusters_path}')
    pd.DataFrame({'uniq_id': df['uniq_id'], 'cluster_label': labels}).to_csv(clusters_path, index=False)

    t1 = time.time()
    report = {
        'rows': int(len(df)),
        'embed_dim': int(X.shape[1]),
        'clusters': int(k),
        'silhouette': sil,
        'time_sec': round(t1 - t0, 2),
        'artifacts': {
            'embeddings': emb_path,
            'faiss_index': faiss_path,
            'metadata': meta_path,
            'cluster_labels': clusters_path,
        },
        'model': args.model,
    }
    safe_json_dump(report, report_path)
    log(f"Training complete in {report['time_sec']}s")


if __name__ == '__main__':
    main()
