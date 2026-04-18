"""
cluster_resumes.py

- load embeddings
- reduce dimensionaity using UMAP
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

# load embeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
EMB_PATH = os.path.join(BASE_DIR, "results","embeddings", "resume_embeddings.npy")

embeddings = np.load(EMB_PATH)
embeddings_raw = embeddings.copy()

print("embedding size:",embeddings.shape)

# Clusterings:

# RAW
data_raw = embeddings_raw

# PCA only
pca = PCA(n_components=50)
data_pca = pca.fit_transform(embeddings_raw)

# UMAP only
umap_only = umap.UMAP(n_neighbors=5,n_components=20,min_dist=0.0, metric="cosine",random_state=42)
data_umap = umap_only.fit_transform(embeddings_raw)

# PCA + UMAP
umap_after_pca = umap.UMAP(n_neighbors=5,n_components=20,min_dist=0.0, metric="cosine",random_state=42)
data_pca_umap = umap_after_pca.fit_transform(data_pca)

# Clustering function
def cluster_data(data):
    clusterer = hdbscan.HDBSCAN( min_cluster_size=6,min_samples=2, cluster_selection_method='leaf', metric="euclidean")
    labels = clusterer.fit_predict(data)

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = list(labels).count(-1)

    return labels, num_clusters, num_noise


datasets = {"raw": data_raw,"pca": data_pca,"umap": data_umap,"pca_umap": data_pca_umap}
results = {}

for name, data in datasets.items():
    print("\n" + "="*50)
    print(f"\nRunning: {name}")
    print("\n" + "="*50)
    labels, num_clusters, num_noise = cluster_data(data)
    results[name] = labels
    print("Clusters:", num_clusters)
    print("Noise points:", num_noise)

    save_path = os.path.join(BASE_DIR, "results", "clusters", f"{name}_labels.npy")
    np.save(save_path, labels)
    
reducer_2d = umap.UMAP(n_components=2, random_state=42)
data_2d = reducer_2d.fit_transform(data)

def plot_clusters(data, labels, title):
    plt.figure(figsize=(6,5))
    plt.scatter(data_2d[:,0], data_2d[:,1], c=labels, cmap="tab10")
    plt.title(title)
    plt.show()


for name, data in datasets.items():
    plot_clusters(data, results[name], f"{name} clustering")
