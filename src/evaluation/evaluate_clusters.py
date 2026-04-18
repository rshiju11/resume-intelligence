"""
evaluate_clusters.py

This file evaluates the quality of clusters generated from resume embeddings.

It computes:
1. Internal evaluation metrics (no labels required):
   - Silhouette Score (higher is better)
   - Davies-Bouldin Index (lower is better)
   - Calinski-Harabasz Score (higher is better)

2. External evaluation metrics (requires true labels):
   - Adjusted Rand Index (ARI)
   - Normalized Mutual Information (NMI)

Important Notes:
- HDBSCAN assigns noise points as label = -1
- These noise points are removed before evaluation
- Evaluation is performed for multiple pipelines:
    RAW, PCA, UMAP, PCA+UMAP

Outputs:
- Prints evaluation scores for each pipeline
- Helps compare which pipeline produces better clustering
"""

import numpy as np
import os

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


# -------------------------------
# INTERNAL CLUSTER EVALUATION
# -------------------------------
def evaluate_internal(data, labels):
    """
    Evaluates clustering using internal metrics.

    Parameters:
    - data: feature matrix used for clustering
    - labels: cluster labels from HDBSCAN

    Returns:
    - dictionary of evaluation scores
    """

    # Remove noise points (label = -1)
    mask = labels != -1
    data_clean = data[mask]
    labels_clean = labels[mask]

    # Ensure at least 2 clusters exist
    if len(set(labels_clean)) < 2:
        print("Not enough clusters for evaluation.")
        return None

    # Compute metrics
    silhouette = silhouette_score(data_clean, labels_clean)
    db_index = davies_bouldin_score(data_clean, labels_clean)
    ch_score = calinski_harabasz_score(data_clean, labels_clean)

    return {
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": db_index,
        "Calinski-Harabasz Score": ch_score
    }


# -------------------------------
# EXTERNAL CLUSTER EVALUATION
# -------------------------------
def evaluate_external(true_labels, cluster_labels):
    """
    Evaluates clustering using ground truth labels.

    Parameters:
    - true_labels: actual resume categories
    - cluster_labels: predicted cluster labels

    Returns:
    - dictionary of evaluation scores
    """

    # Remove noise points
    mask = cluster_labels != -1
    true_clean = true_labels[mask]
    cluster_clean = cluster_labels[mask]

    # Compute metrics
    ari = adjusted_rand_score(true_clean, cluster_clean)
    nmi = normalized_mutual_info_score(true_clean, cluster_clean)

    return {
        "Adjusted Rand Index (ARI)": ari,
        "Normalized Mutual Info (NMI)": nmi
    }


# -------------------------------
# LOAD DATA
# -------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Load embeddings (same as clustering step)
EMB_PATH = os.path.join(BASE_DIR, "results", "embeddings", "resume_embeddings.npy")
embeddings = np.load(EMB_PATH)

# Load cluster labels (you should have saved these)
labels_raw = np.load(os.path.join(BASE_DIR, "results", "clusters", "raw_labels.npy"))
labels_pca = np.load(os.path.join(BASE_DIR, "results", "clusters", "pca_labels.npy"))
labels_umap = np.load(os.path.join(BASE_DIR, "results", "clusters", "umap_labels.npy"))
labels_pca_umap = np.load(os.path.join(BASE_DIR, "results", "clusters", "pca_umap_labels.npy"))

# OPTIONAL: Load true labels if available
# Example: true_labels = np.load("true_labels.npy")
# If you don't have labels, skip external evaluation
true_labels_path = os.path.join(BASE_DIR, "data", "processed", "true_labels.npy")

if os.path.exists(true_labels_path):
    true_labels = np.load(true_labels_path)
else:
    true_labels = None


# -------------------------------
# EVALUATION FOR EACH PIPELINE
# -------------------------------

print("\n===== INTERNAL EVALUATION =====\n")

print("RAW:")
print(evaluate_internal(embeddings, labels_raw), "\n")

print("PCA:")
print(evaluate_internal(embeddings, labels_pca), "\n")

print("UMAP:")
print(evaluate_internal(embeddings, labels_umap), "\n")

print("PCA + UMAP:")
print(evaluate_internal(embeddings, labels_pca_umap), "\n")


# -------------------------------
# EXTERNAL EVALUATION (OPTIONAL)
# -------------------------------

if true_labels is not None:
    print("\n===== EXTERNAL EVALUATION =====\n")

    print("RAW:")
    print(evaluate_external(true_labels, labels_raw), "\n")

    print("PCA:")
    print(evaluate_external(true_labels, labels_pca), "\n")

    print("UMAP:")
    print(evaluate_external(true_labels, labels_umap), "\n")

    print("PCA + UMAP:")
    print(evaluate_external(true_labels, labels_pca_umap), "\n")

else:
    print("\n(No ground truth labels found — skipping external evaluation)\n")


# -------------------------------
# END OF FILE
# -------------------------------
"""
This script completes the evaluation phase of the resume clustering pipeline.

Key Takeaways:
- Internal metrics evaluate structure of clusters
- External metrics evaluate alignment with real-world categories
- Combined, these provide a strong justification for clustering quality

This step is essential to transform the project from a simple pipeline
into a research-quality analysis.
"""