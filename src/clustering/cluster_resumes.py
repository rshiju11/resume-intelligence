"""
cluster_resumes.py

- load embeddings
- reduce dimensionality using PCA + UMAP
- form clusters using HDBSCAN algorithm
- save cluster labels
- visualize 2d structure
"""

import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# -------------------------------
# PLOT FUNCTION
# -------------------------------
def plot_clusters(data_2d, labels, title):
    plt.figure(figsize=(6,5))
    plt.scatter(data_2d[:,0], data_2d[:,1], c=labels, cmap="tab20", s=20)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.show()


# LOAD EMBEDDINGS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
EMB_PATH = os.path.join(BASE_DIR, "results","embeddings", "resume_embeddings.npy")

embeddings = np.load(EMB_PATH)
print("embedding size:", embeddings.shape)


# -------------------------------
# DATA PREPARATION
# -------------------------------

# RAW
data_raw = embeddings

# PCA
pca = PCA(n_components=50, random_state=42)
data_pca = pca.fit_transform(embeddings)


# -------------------------------
# UMAP (FINAL BEST SETTINGS)
# -------------------------------

umap_only = umap.UMAP(
    n_neighbors=10,
    n_components=10,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)
data_umap = umap_only.fit_transform(embeddings)


umap_after_pca = umap.UMAP(
    n_neighbors=10,
    n_components=10,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)
data_pca_umap = umap_after_pca.fit_transform(data_pca)

# HDBSCAN (FINAL SETTINGS)
def cluster_data(data):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=1,
        metric="euclidean"
    )

    labels = clusterer.fit_predict(data)

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = list(labels).count(-1)

    return labels, num_clusters, num_noise


# DATASETS
datasets = {
    "raw": data_raw,
    "pca": data_pca,
    "umap": data_umap,
    "pca_umap": data_pca_umap
}

results = {}

# MAIN LOOP
for name, data in datasets.items():

    print("\n" + "="*50)
    print(f"\nRunning: {name}")
    print("\n" + "="*50)

    # CLUSTER
    labels, num_clusters, num_noise = cluster_data(data)
    results[name] = labels

    print("Clusters:", num_clusters)
    print("Noise points:", num_noise)

    # SAVE LABELS
    save_path = os.path.join(BASE_DIR, "results", "clusters", f"{name}_labels.npy")
    np.save(save_path, labels)

    # 2D VISUALIZATION
    reducer_2d = umap.UMAP(n_components=2, random_state=42)
    data_2d = reducer_2d.fit_transform(data)

    plot_clusters(data_2d, labels, f"{name} clustering")