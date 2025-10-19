from fastapi import APIRouter
import os
import json
import pandas as pd
from pathlib import Path

router = APIRouter()


@router.get("/api/analytics")
def analytics():
    # use the backend package location to build a stable path to backend/data
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    cache_dir = os.path.join(base_dir, "data")
    cache = os.path.join(cache_dir, "analytics.json")
    # if cache exists, return it
    if os.path.exists(cache):
        # read cache as text and replace invalid JSON tokens (NaN/Infinity) with null
        with open(cache, "r") as f:
            txt = f.read()
        clean = txt.replace('NaN', 'null').replace('nan', 'null').replace('Infinity', 'null').replace('-Infinity', 'null')
        # try to parse to confirm it's valid JSON; if parsing fails, return cleaned text anyway
        try:
            _ = json.loads(clean)
        except Exception:
            pass
        from fastapi import Response
        return Response(content=clean, media_type='application/json')
    # compute from dataset
    path = os.getenv('DATASET_PATH', os.path.join(os.getcwd(), "backend", "data", "mock_furniture.csv"))
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    # simple aggregations
    # Clean/parse price column: strip currency symbols/commas and convert to numeric
    if 'price' in df.columns:
        # coerce everything to string, remove non-number characters except dot and minus, then convert
        df['price_clean'] = df['price'].astype(str).str.replace(r"[^0-9\.-]", '', regex=True)
        df['price_clean'] = pd.to_numeric(df['price_clean'], errors='coerce')
    else:
        df['price_clean'] = pd.NA

    cat_counts = df['categories'].value_counts().reset_index().rename(columns={"index": "category", "categories": "count"}).to_dict(orient='records')
    avg_price = df.groupby('categories')['price_clean'].mean().reset_index().rename(columns={'price_clean':'avg_price'}).to_dict(orient='records')
    top_brands = df['brand'].value_counts().reset_index().rename(columns={"index": "brand", "brand": "count"}).head(10).to_dict(orient='records')
    materials = df['material'].value_counts().reset_index().rename(columns={"index": "material", "material": "count"}).to_dict(orient='records')
    out = {
        'categoryCounts': cat_counts,
        'avgPriceByCategory': avg_price,
        'brandTop': top_brands,
        'materials': materials
    }
    # Build output
    out = {
        'categoryCounts': cat_counts,
        'avgPriceByCategory': avg_price,
        'brandTop': top_brands,
        'materials': materials
    }

    # Recursive sanitizer to make everything JSON-serializable
    import math
    import numpy as _np

    def sanitize_value(v):
        # None
        if v is None:
            return None
        # pandas NA and numpy NaN
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        # numpy scalar
        if isinstance(v, (_np.floating, _np.integer)):
            return v.item()
        # numpy array
        if isinstance(v, _np.ndarray):
            return [sanitize_value(x) for x in v.tolist()]
        # floats: check NaN/inf
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        # dict
        if isinstance(v, dict):
            return {k: sanitize_value(val) for k, val in v.items()}
        # list/tuple
        if isinstance(v, (list, tuple)):
            return [sanitize_value(x) for x in v]
        # other numpy types
        if hasattr(v, 'item'):
            try:
                return v.item()
            except Exception:
                pass
        return v

    sanitized_out = sanitize_value(out)

    # ensure the cache directory exists before writing
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    # write cache using standard json (this may write NaN as null-safe token if present)
    with open(cache, 'w') as f:
        json.dump(sanitized_out, f)

    # Pre-serialize to JSON to avoid Starlette JSON re-serialization issues with NaN
    from fastapi import Response
    json_text = json.dumps(sanitized_out, allow_nan=True)
    # Replace NaN/Infinity tokens with null for strict JSON consumers
    json_text = json_text.replace('NaN', 'null').replace('nan', 'null').replace('Infinity', 'null').replace('-Infinity', 'null')
    return Response(content=json_text, media_type='application/json')
