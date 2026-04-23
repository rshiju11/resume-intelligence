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

# PLOT FUNCTION
def plot_clusters(data_2d, labels, title):
    plt.figure(figsize=(6,5))
    labels = np.array(labels)

    # Separate noise and clusters
    noise_mask = labels == -1
    cluster_mask = labels != -1

     # Plot clusters
    plt.scatter(data_2d[cluster_mask,0], data_2d[cluster_mask,1],
                c=labels[cluster_mask], cmap="tab20", s=20)

    # Plot noise in gray
    plt.scatter(data_2d[noise_mask,0], data_2d[noise_mask,1],
                color="gray", s=20, label="Noise")

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.show()


# LOAD EMBEDDINGS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
EMB_PATH = os.path.join(BASE_DIR, "results","embeddings", "resume_embeddings.npy")

embeddings = np.load(EMB_PATH)
print("embedding size:", embeddings.shape)

# DATA PREPARATION

# RAW
data_raw = embeddings

# PCA
pca = PCA(n_components=50, random_state=42)
data_pca = pca.fit_transform(embeddings)

# UMAP

umap_only = umap.UMAP(n_neighbors=10,n_components=10,min_dist=0.1, metric="cosine",random_state=42)
data_umap = umap_only.fit_transform(embeddings)


umap_after_pca = umap.UMAP( n_neighbors=10,n_components=10,min_dist=0.1,metric="cosine", random_state=42)
data_pca_umap = umap_after_pca.fit_transform(data_pca)

# HDBSCAN 
def cluster_data(data):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1, metric="euclidean" )

    labels = clusterer.fit_predict(data)

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = list(labels).count(-1)

    return labels, num_clusters, num_noise


# DATASETS
datasets = {"Raw": data_raw, "PCA": data_pca, "UMAP": data_umap, "PCA+UMAP": data_pca_umap }

results = {}

# LOAD FILENAMES
file_path = os.path.join(BASE_DIR, "results", "embeddings", "filenames.txt")

with open(file_path, "r") as f:
    filenames = [line.strip() for line in f.readlines()]

# MAIN LOOP
for name, data in datasets.items():

    print("\n" + "="*50)
    print(f"\nRunning: {name}")
    print("\n" + "="*50)

    # CLUSTER
    labels, num_clusters, num_noise = cluster_data(data)
    
    print("\nCluster contents:")

    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(filenames[idx])

    for cluster_id, files in clusters.items():
        if cluster_id == -1:
            print(f"\nNoise ({len(files)} resumes):")
        else:
            print(f"\nCluster {cluster_id} ({len(files)} resumes):")

        for file in files[:5]:  # show only first 5 to keep it clean
            print("  ", file)

    results[name] = labels

    print("Clusters:", num_clusters)
    print("Noise points:", num_noise)

    # SAVE LABELS
    save_path = os.path.join(BASE_DIR, "results", "clusters", f"{name}_labels.npy")
    np.save(save_path, labels)

    output_path = os.path.join(BASE_DIR, "results", "clusters", f"{name}_clusters.txt")

    with open(output_path, "w") as f:
        for cluster_id in sorted(clusters.keys()):
            files = clusters[cluster_id]
            if cluster_id == -1:
                f.write(f"\nNoise ({len(files)} resumes):\n")
            else:
                f.write(f"\nCluster {cluster_id} ({len(files)} resumes):\n")

            for file in files:
                f.write(f"  {file}\n")

    # 2D VISUALIZATION
    reducer_2d = umap.UMAP(n_components=2, random_state=42)
    data_2d = reducer_2d.fit_transform(data)

    plot_clusters(data_2d, labels, f"{name} Clusters Plot")

    