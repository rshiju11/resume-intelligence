"""
evaluate_clusters.py

Evaluates clustering quality using:
- Internal metrics
- External metrics (if labels exist)
- Semantic similarity
- Cluster vs label distribution
"""

import numpy as np
import os
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

from sklearn.metrics.pairwise import cosine_similarity


# INTERNAL EVALUATION
def evaluate_internal(data, labels):

    mask = labels != -1
    data_clean = data[mask]
    labels_clean = labels[mask]

    if len(set(labels_clean)) < 2:
        print("Not enough clusters for evaluation.")
        return None

    return {
        "Silhouette Score": silhouette_score(data_clean, labels_clean),
        "Davies-Bouldin Index": davies_bouldin_score(data_clean, labels_clean),
        "Calinski-Harabasz Score": calinski_harabasz_score(data_clean, labels_clean)
    }

# EXTERNAL EVALUATION
def evaluate_external(true_labels, cluster_labels):

    mask = cluster_labels != -1
    true_clean = true_labels[mask]
    cluster_clean = cluster_labels[mask]

    return {
        "Adjusted Rand Index (ARI)": adjusted_rand_score(true_clean, cluster_clean),
        "Normalized Mutual Info (NMI)": normalized_mutual_info_score(true_clean, cluster_clean)
    }


# SEMANTIC VALIDATION
def semantic_similarity_analysis(embeddings, labels):

    sim_matrix = cosine_similarity(embeddings)

    within = []
    between = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):

            if labels[i] == -1 or labels[j] == -1:
                continue

            if labels[i] == labels[j]:
                within.append(sim_matrix[i][j])
            else:
                between.append(sim_matrix[i][j])

    print("\n===== SEMANTIC VALIDATION =====")
    print("Avg within-cluster similarity:", np.mean(within))
    print("Avg between-cluster similarity:", np.mean(between))

# CLUSTER vs TRUE LABEL TABLE
def cluster_label_distribution(true_labels, cluster_labels):

    mask = cluster_labels != -1
    true_clean = true_labels[mask]
    cluster_clean = cluster_labels[mask]

    df = pd.DataFrame({
        "Cluster": cluster_clean,
        "True Label": true_clean
    })

    table = pd.crosstab(df["Cluster"], df["True Label"])

    print("\n===== CLUSTER vs LABEL DISTRIBUTION =====")
    print(table)

# LOAD DATA
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

EMB_PATH = os.path.join(BASE_DIR, "results", "embeddings", "resume_embeddings.npy")
embeddings = np.load(EMB_PATH)

# IMPORTANT: load correct datasets
labels_raw = np.load(os.path.join(BASE_DIR, "results", "clusters", "raw_labels.npy"))
labels_pca = np.load(os.path.join(BASE_DIR, "results", "clusters", "pca_labels.npy"))
labels_umap = np.load(os.path.join(BASE_DIR, "results", "clusters", "umap_labels.npy"))
labels_pca_umap = np.load(os.path.join(BASE_DIR, "results", "clusters", "pca_umap_labels.npy"))

# Load reduced data again for correct evaluation
from sklearn.decomposition import PCA
import umap

pca = PCA(n_components=50, random_state=42)
data_pca = pca.fit_transform(embeddings)

data_umap = umap.UMAP(n_neighbors=10, n_components=10, min_dist=0.1, metric="cosine", random_state=42).fit_transform(embeddings)
data_pca_umap = umap.UMAP(n_neighbors=10, n_components=10, min_dist=0.1, metric="cosine", random_state=42).fit_transform(data_pca)


# INTERNAL EVALUATION (FIXED)

print("\n===== INTERNAL EVALUATION =====\n")

print("RAW:")
print(evaluate_internal(embeddings, labels_raw), "\n")

print("PCA:")
print(evaluate_internal(data_pca, labels_pca), "\n")

print("UMAP:")
print(evaluate_internal(data_umap, labels_umap), "\n")

print("PCA + UMAP:")
print(evaluate_internal(data_pca_umap, labels_pca_umap), "\n")


# EXTERNAL EVALUATION
true_labels_path = os.path.join(BASE_DIR, "data", "processed", "true_labels.npy")

if os.path.exists(true_labels_path):

    true_labels = np.load(true_labels_path)

    print("\n===== EXTERNAL EVALUATION =====\n")

    print("PCA + UMAP:")
    print(evaluate_external(true_labels, labels_pca_umap), "\n")

    cluster_label_distribution(true_labels, labels_pca_umap)

else:
    print("\n(No ground truth labels found — skipping external evaluation)\n")

# SEMANTIC VALIDATION
semantic_similarity_analysis(embeddings, labels_pca_umap)