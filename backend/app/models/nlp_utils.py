import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter

def fit_product_clusters(embeddings: np.ndarray, n_clusters: int = 12):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(embeddings)
    score = None
    try:
        score = silhouette_score(embeddings, labels)
    except Exception:
        score = None
    return labels, {'silhouette': score}

def get_cluster_name(label, top_titles):
    # heuristic: look for common words in top_titles
    words = Counter()
    for t in top_titles:
        for w in t.lower().split():
            if len(w) > 3:
                words[w] += 1
    if not words:
        return f"Cluster {label}"
    common = words.most_common(2)
    name = " ".join([w for w, _ in common]).title()
    return name

def get_cluster_stats(df):
    if 'cluster_label' not in df.columns:
        return {}
    counts = df['cluster_label'].value_counts().to_dict()
    return counts
