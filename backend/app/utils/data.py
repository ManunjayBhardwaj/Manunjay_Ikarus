import pandas as pd
import os
import ast
import uuid

# canonical column names we want to keep (map common variants)
KEEP = ['uniq_id', 'title', 'brand', 'description', 'price', 'categories', 'images', 'manufacturer', 'package_dimensions', 'country_of_origin', 'material', 'color']

COL_ALIASES = {
    'package dimensions': 'package_dimensions',
    'package_dimensions': 'package_dimensions',
    'package-dimensions': 'package_dimensions',
    'country of origin': 'country_of_origin',
    'country_of_origin': 'country_of_origin',
}


def _parse_price(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    # remove common currency symbols and commas
    for ch in ['₹', '$', '€', '£', 'Rs', ',']:
        s = s.replace(ch, '')
    try:
        return float(s)
    except Exception:
        return None


def _parse_list_field(x):
    # Handles stringified python lists or simple comma-separated strings
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(v).strip() for v in val]
    except Exception:
        pass
    # fallback split on comma
    return [p.strip() for p in s.split(',') if p.strip()]


def load_and_clean(path):
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    cols = {c: c for c in df.columns}
    for c in list(cols.keys()):
        low = c.lower()
        if low in COL_ALIASES:
            cols[c] = COL_ALIASES[low]
        else:
            # normalize spaces/dashes
            k = low.replace(' ', '_').replace('-', '_')
            if k in KEEP:
                cols[c] = k
    df = df.rename(columns=cols)

    # ensure keep columns exist
    for c in KEEP:
        if c not in df.columns:
            df[c] = None

    # fill defaults
    df['title'] = df['title'].fillna('Unknown')
    df['description'] = df['description'].fillna('')
    df['brand'] = df['brand'].fillna('Unknown')

    # parse price
    df['price'] = df['price'].apply(_parse_price)

    # parse categories/images as lists and extract primary image
    df['categories'] = df['categories'].apply(_parse_list_field)
    df['images'] = df['images'].apply(_parse_list_field)
    df['image'] = df['images'].apply(lambda lst: lst[0] if lst else None)

    # ensure uniq_id
    def make_uid(x):
        if pd.isna(x) or not str(x).strip():
            return str(uuid.uuid4())
        return str(x)

    df['uniq_id'] = df.get('uniq_id', df.get('id', None)).apply(make_uid)

    # return canonical subset
    return df[KEEP + ['image']]
