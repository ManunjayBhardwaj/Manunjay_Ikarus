from fastapi import APIRouter, HTTPException
import os
import json
from typing import Optional

router = APIRouter()


@router.get("/api/sample-data")
def sample_data(n: Optional[int] = 5):
    # build package-relative path to storage metadata
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    storage_dir = os.path.join(base_dir, "backend", "storage")
    meta_path = os.path.join(storage_dir, "metadata.json")

    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail=f"metadata.json not found at {meta_path}")

    with open(meta_path, "r") as f:
        try:
            j = json.load(f)
        except Exception:
            # fallback: read as lines and try to parse as JSON per-line dict
            f.seek(0)
            lines = f.readlines()
            j = {}
            for i, line in enumerate(lines):
                line = line.strip().rstrip(',')
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    j.update(obj if isinstance(obj, dict) else {})
                except Exception:
                    # skip unparsable lines
                    continue

    # j is expected to be a dict of id -> metadata
    if isinstance(j, dict):
        vals = list(j.values())[:n]
        # ensure basic sanitization for numpy/pandas types (simple types only)
        def simple_sanitize(v):
            try:
                import math
                import numpy as _np
                if v is None:
                    return None
                if isinstance(v, (_np.floating, _np.integer)):
                    return v.item()
                if isinstance(v, _np.ndarray):
                    return v.tolist()
                if isinstance(v, float):
                    if math.isnan(v) or math.isinf(v):
                        return None
                    return v
            except Exception:
                pass
            if isinstance(v, dict):
                return {k: simple_sanitize(val) for k, val in v.items()}
            if isinstance(v, (list, tuple)):
                return [simple_sanitize(x) for x in v]
            return v

        sanitized = [simple_sanitize(x) for x in vals]
        return sanitized

    # if it's a list, return first n
    if isinstance(j, list):
        return j[:n]

    raise HTTPException(status_code=500, detail="Unable to parse metadata.json")
