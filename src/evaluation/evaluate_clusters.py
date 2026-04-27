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
from sklearn.metrics import ( silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score,normalized_mutual_info_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import umap

def evaluate_internal(data, labels):
    mask = labels != -1
    data_clean = data[mask]
    labels_clean = labels[mask]

    if len(set(labels_clean)) < 2:
        print("Not enough clusters for evaluation.")
        return None

    return {"Silhouette Score": silhouette_score(data_clean, labels_clean),
        "Davies-Bouldin Index": davies_bouldin_score(data_clean, labels_clean),
        "Calinski-Harabasz Score": calinski_harabasz_score(data_clean, labels_clean) }

def evaluate_external(true_labels, cluster_labels):
    mask = cluster_labels != -1
    true_clean = true_labels[mask]
    cluster_clean = cluster_labels[mask]

    return { "Adjusted Rand Index (ARI)": adjusted_rand_score(true_clean, cluster_clean),
        "Normalized Mutual Info (NMI)": normalized_mutual_info_score(true_clean, cluster_clean) }

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

    print("\nEemantic Validation Results:")
    print("Avg within-cluster similarity:", np.mean(within))
    print("Avg between-cluster similarity:", np.mean(between))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

EMB_PATH = os.path.join(BASE_DIR, "results", "embeddings", "resume_embeddings.npy")
embeddings = np.load(EMB_PATH)
labels_raw = np.load(os.path.join(BASE_DIR, "results", "clusters", "raw_labels.npy"))
labels_pca = np.load(os.path.join(BASE_DIR, "results", "clusters", "pca_labels.npy"))
labels_umap = np.load(os.path.join(BASE_DIR, "results", "clusters", "umap_labels.npy"))
labels_pca_umap = np.load(os.path.join(BASE_DIR, "results", "clusters", "pca_umap_labels.npy"))

pca = PCA(n_components=50, random_state=42)
data_pca = pca.fit_transform(embeddings)

data_umap = umap.UMAP(n_neighbors=10, n_components=10, min_dist=0.1, metric="cosine", random_state=42).fit_transform(embeddings)
data_pca_umap = umap.UMAP(n_neighbors=10, n_components=10, min_dist=0.1, metric="cosine", random_state=42).fit_transform(data_pca)


# Internal metrics

print("\nInternal evaluation\n")

print("RAW:")
print(evaluate_internal(embeddings, labels_raw), "\n")

print("PCA:")
print(evaluate_internal(data_pca, labels_pca), "\n")

print("UMAP:")
print(evaluate_internal(data_umap, labels_umap), "\n")

print("PCA + UMAP:")
print(evaluate_internal(data_pca_umap, labels_pca_umap), "\n")


# External Eval,
true_labels_path = os.path.join(BASE_DIR, "data", "processed", "true_labels.npy")

if os.path.exists(true_labels_path):

    true_labels = np.load(true_labels_path)

    print("\nExternal Evaluation\n")

    print("PCA + UMAP:")
    print(evaluate_external(true_labels, labels_pca_umap), "\n")

else:
    print("\n(No ground truth labels found -- skipping external evaluation)\n")


semantic_similarity_analysis(embeddings, labels_pca_umap)