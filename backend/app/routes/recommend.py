from fastapi import APIRouter
from pydantic import BaseModel
import asyncio
import math
import numpy as np

router = APIRouter()


class RecommendRequest(BaseModel):
    query: str
    k: int = 5


@router.post("/api/recommend")
async def recommend(req: RecommendRequest):
    # Lazy import heavy modules so app can start without ML libs installed
    from ..models.recommender import recommend_by_query
    from ..models.genai import generate_marketing_copy

    items = recommend_by_query(req.query, k=req.k)

    async def make_copy(it):
        return generate_marketing_copy(it)

    copies = await asyncio.gather(*[make_copy(it) for it in items])
    for i, c in enumerate(copies):
        items[i]['generated_copy'] = c
    # Ensure each item has a top-level image_url for frontend convenience
    for it in items:
        # if image_url already present and truthy, keep it
        img = it.get('image_url') or it.get('image')
        if not img:
            meta = it.get('metadata') or {}
            # check common metadata image fields
            img = meta.get('image') if isinstance(meta, dict) else None
            if not img:
                imgs = meta.get('images') if isinstance(meta, dict) else None
                if isinstance(imgs, (list, tuple)) and len(imgs) > 0:
                    img = imgs[0]
        if img:
            it['image_url'] = img
    # sanitize items to ensure JSON serializability (convert numpy types, NaN, inf)
    def sanitize_value(v):
        # numpy scalar
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        # numpy arrays -> lists
        if isinstance(v, np.ndarray):
            return v.tolist()
        # floats: NaN or infinite are not JSON serializable
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        # dict -> sanitize recursively
        if isinstance(v, dict):
            return {k: sanitize_value(val) for k, val in v.items()}
        # list/tuple
        if isinstance(v, (list, tuple)):
            return [sanitize_value(x) for x in v]
        return v

    sanitized = [sanitize_value(it) for it in items]
    return sanitized
