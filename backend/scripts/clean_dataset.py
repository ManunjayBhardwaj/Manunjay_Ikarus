#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean furniture dataset for the Product Recommendation web app.

Outputs:
- cleaned_furniture.csv
- cleaned_furniture.parquet (optional via --parquet)
- analytics.json
- schema_report.json

Usage:
  python clean_dataset.py --input intern_data_ikarus.csv --outdir backend/data --parquet
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd


ASSIGNMENT_COLUMNS = [
    "title",
    "brand",
    "description",
    "price",
    "categories",
    "images",
    "manufacturer",
    "package dimensions",
    "country_of_origin",
    "material",
    "color",
    "uniq_id",
]

REQUIRED_MINIMAL = ["uniq_id", "title", "description", "price", "categories", "images", "brand"]


def log(msg: str):
    print(f"[clean] {msg}", flush=True)


def coalesce_columns(df: pd.DataFrame, candidates: list[str], target: str) -> pd.Series:
    """Coalesce the first non-null/non-empty from candidate columns to a target Series."""
    s = pd.Series([None] * len(df))
    for c in candidates:
        if c in df.columns:
            v = df[c].astype(str).str.strip()
            v = v.replace({"nan": None, "None": None, "": None})
            s = s.fillna(v)
    return s


def normalize_price(value) -> float | None:
    """
    Convert a price string like '₹5,999.00', '5999', '$89.90', 'Rs 2,499', '4.2k' into float (INR-like).
    Heuristics:
      - strip currency symbols and commas
      - handle 'k' or 'K' as * 1000
    """
    if pd.isna(value):
        return None
    s = str(value)
    if not s or s.lower() in {"nan", "none"}:
        return None

    s = s.replace(",", "").strip()
    # remove currency words/symbols
    s = re.sub(r"(rs\.?|₹|\$|usd|inr|eur|gbp)", "", s, flags=re.IGNORECASE).strip()

    # handle k/K suffix
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*[kK]", s)
    if m:
        try:
            return float(m.group(1)) * 1000.0
        except Exception:
            return None

    # extract first float number
    m = re.search(r"([0-9]*\.?[0-9]+)", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def pick_primary_image(images_val: str | None) -> str | None:
    """
    From 'images' field that may contain:
      - single URL
      - JSON-like list
      - pipe/semicolon/comma separated URLs
    Return the first plausible URL (http/https) or None.
    """
    if images_val is None or (isinstance(images_val, float) and math.isnan(images_val)):
        return None
    s = str(images_val).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None

    # Try JSON-like list
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            arr = ast.literal_eval(s)
            if isinstance(arr, (list, tuple)) and arr:
                s = str(arr[0])
            else:
                s = ""
        except Exception:
            pass

    # Split by common delimiters to find URLs
    parts = re.split(r"[|\;\,\s]+", s)
    for p in parts:
        p = p.strip().strip("'\"")
        if p.startswith("http://") or p.startswith("https://"):
            return p

    # If still not found but looks like a path, return first token
    return parts[0].strip().strip("'\"") if parts else None


def normalize_categories(val: str | None) -> list[str]:
    """
    Normalize categories to a list of strings.
      - split by '>', '/', '|', ',', ';'
      - strip and title-case lightly
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    s = str(val)
    if not s or s.lower() in {"nan", "none"}:
        return []
    tokens = re.split(r"[>/\|\;\,]+", s)
    tokens = [t.strip() for t in tokens if t.strip()]
    # keep case reasonable
    return tokens


def normalize_text(text: str | None, fallback: str = "Unknown") -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return fallback
    s = str(text).strip()
    return s if s else fallback


def standardize_color(val: str | None) -> str | None:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    # basic normalization
    s = s.lower()
    mapping = {
        "blk": "black", "wht": "white", "gry": "grey", "gr": "green", "brn": "brown",
        "rd": "red", "bl": "blue", "ylw": "yellow"
    }
    return mapping.get(s, s)


def standardize_material(val: str | None) -> str | None:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip().lower()
    if not s:
        return None
    replacements = {
        "wooden": "wood",
        "engineered wood": "engineered-wood",
        "mango wood": "wood",
        "solid wood": "wood",
        "ply": "plywood",
        "mdf board": "mdf",
    }
    return replacements.get(s, s)


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all assignment columns exist; create empty if missing."""
    for col in ASSIGNMENT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def make_schema_report(df: pd.DataFrame) -> dict:
    rep = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "rows": int(len(df)),
        "columns": {},
    }
    for c in ASSIGNMENT_COLUMNS:
        col = df[c] if c in df.columns else pd.Series(dtype=object)
        rep["columns"][c] = {
            "present": c in df.columns,
            "non_null": int(col.notna().sum()),
            "null": int(col.isna().sum()),
            "unique": int(col.nunique(dropna=True)),
            "example_values": col.dropna().astype(str).head(5).tolist(),
            "dtype": str(col.dtype) if c in df.columns else None,
        }
    return rep


def compute_analytics(df: pd.DataFrame) -> dict:
    # Flatten categories for counts
    cat_counts = Counter()
    for cats in df["categories_norm"]:
        for c in cats:
            cat_counts[c] += 1

    # Average price by top-level category (first token)
    topcat_prices = defaultdict(list)
    for cats, p in zip(df["categories_norm"], df["price_num"]):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            continue
        top = cats[0] if cats else "Uncategorized"
        topcat_prices[top].append(float(p))
    avg_price_by_cat = [{"category": k, "avg_price": float(np.mean(v))} for k, v in topcat_prices.items()]

    # Brand counts
    brand_counts = Counter(df["brand_norm"].fillna("Unknown").tolist())

    # Material distribution
    material_counts = Counter(df["material_std"].fillna("unknown").tolist())

    return {
        "categoryCounts": [{"category": k, "count": v} for k, v in cat_counts.most_common()],
        "avgPriceByCategory": sorted(avg_price_by_cat, key=lambda x: x["avg_price"], reverse=True),
        "brandTop": [{"brand": k, "count": v} for k, v in brand_counts.most_common(20)],
        "materials": [{"material": k, "count": v} for k, v in material_counts.most_common()],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV (e.g., intern_data_ikarus.csv)")
    parser.add_argument("--outdir", default="backend/data", help="Output directory")
    parser.add_argument("--outfile", default="cleaned_furniture.csv", help="Output CSV filename")
    parser.add_argument("--parquet", action="store_true", help="Also write Parquet")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, args.outfile)
    out_parquet = os.path.join(args.outdir, os.path.splitext(args.outfile)[0] + ".parquet")
    analytics_path = os.path.join(args.outdir, "analytics.json")
    schema_path = os.path.join(args.outdir, "schema_report.json")

    log(f"Loading: {args.input}")
    try:
        df = pd.read_csv(args.input, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(args.input, encoding="latin-1", low_memory=False)

    # Ensure required columns exist (create if missing)
    df = ensure_columns(df)

    # Coalesce/normalize core fields
    # uniq_id
    if df["uniq_id"].isna().all():
        # fallback: generate deterministic IDs from title+brand+price row index
        df["uniq_id"] = (
            df["title"].fillna("").astype(str).str.slice(0, 16).str.replace(r"\s+", "_", regex=True)
            + "_" + df.index.astype(str)
        )
        log("uniq_id was missing; generated fallback IDs.")
    else:
        df["uniq_id"] = df["uniq_id"].astype(str).str.strip()

    # title
    df["title_norm"] = df["title"].apply(lambda x: normalize_text(x, "Unknown Title"))
    # brand
    df["brand_norm"] = df["brand"].apply(lambda x: normalize_text(x, "Unknown"))
    # description
    df["description_norm"] = df["description"].apply(lambda x: normalize_text(x, "No description available."))

    # price
    df["price_num"] = df["price"].apply(normalize_price)

    # categories
    df["categories_norm"] = df["categories"].apply(normalize_categories)

    # images -> primary_image
    df["primary_image"] = df["images"].apply(pick_primary_image)

    # material / color
    df["material_std"] = df["material"].apply(standardize_material)
    df["color_std"] = df["color"].apply(standardize_color)

    # country_of_origin
    df["country_of_origin_norm"] = df["country_of_origin"].apply(lambda x: normalize_text(x, "Unknown"))

    # package dimensions -> keep as raw string; try to standardize simple patterns
    df["package_dimensions_norm"] = df["package dimensions"].astype(str).str.strip()

    # manufacturer
    df["manufacturer_norm"] = df["manufacturer"].apply(lambda x: normalize_text(x, "Unknown"))

    # Drop exact duplicates by uniq_id first, then by (title, brand, price)
    before = len(df)
    df = df.sort_values(by=["uniq_id"]).drop_duplicates(subset=["uniq_id"], keep="first")
    after_uid = len(df)
    df = df.drop_duplicates(subset=["title_norm", "brand_norm", "price_num"], keep="first")
    after_tbp = len(df)
    log(f"Dedup: {before} -> {after_uid} (uniq_id) -> {after_tbp} (title,brand,price)")

    # Sanity filter: keep rows with at least minimal information
    df = df[~df["title_norm"].isna()]
    df = df[~df["description_norm"].isna()]

    # Final selected columns for the app (keeping originals where useful)
    out_cols = [
        "uniq_id",
        "title_norm",
        "brand_norm",
        "description_norm",
        "price_num",
        "categories_norm",
        "primary_image",
        "manufacturer_norm",
        "package_dimensions_norm",
        "country_of_origin_norm",
        "material_std",
        "color_std",
        # keep originals for reference (optional)
        "title", "brand", "description", "price", "categories", "images", "manufacturer",
        "package dimensions", "country_of_origin", "material", "color"
    ]

    # Some datasets can miss columns; filter to existing
    out_cols = [c for c in out_cols if c in df.columns]

    # Write cleaned CSV
    log(f"Writing cleaned CSV: {out_csv}")
    df[out_cols].to_csv(out_csv, index=False)

    # Optional Parquet
    if args.parquet:
        log(f"Writing Parquet: {out_parquet}")
        try:
            df[out_cols].to_parquet(out_parquet, index=False)
        except Exception as e:
            log(f"Parquet write failed (install pyarrow or fastparquet): {e}")

    # Analytics JSON (for /api/analytics)
    log(f"Computing analytics -> {analytics_path}")
    analytics = compute_analytics(df)
    with open(analytics_path, "w", encoding="utf-8") as f:
        json.dump(analytics, f, ensure_ascii=False, indent=2)

    # Schema report
    log(f"Writing schema report -> {schema_path}")
    report = make_schema_report(df)
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Quick console summary
    log("Summary:")
    log(f"  Rows cleaned: {len(df)}")
    log(f"  Example categories: {list({(c[0] if c else 'Uncategorized') for c in df['categories_norm']})[:8]}")
    log("Done.")


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None  # silence chained assignment warnings
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)
