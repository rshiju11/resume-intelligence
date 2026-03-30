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
from sklearn.preprocessing import normalize
import os

# load embeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
EMB_PATH = os.path.join(BASE_DIR, "results","embeddings", "resume_embeddings.npy")
SAVE_PATH = os.path.join(BASE_DIR, "results","clusters", "cluster_labels.npy")

embeddings = np.load(EMB_PATH)

print("embedding size: ", embeddings.shape)

#  UMAP reduction

reducer = umap.UMAP(n_neighbors=15, n_components=15, metric="cosine", random_state=42)
reduced_embeddings = reducer.fit_transform(embeddings)
print("reduced embedding size:", reduced_embeddings.shape)

# HDBSCAN cluster
clusterer = hdbscan.HDBSCAN(min_cluster_size=4, metric="euclidean", min_samples=3)
labels = clusterer.fit_predict(reduced_embeddings)

np.save(SAVE_PATH, labels)

print("Cluster labels saved.")
print("Unique labels:", set(labels))

# number of clusters and noise
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
num_noise = list(labels).count(-1)
print("No. of clusters:",num_clusters)
print("No. of noise:", num_noise)


# 2D visualisation
reducer_2d = umap.UMAP(n_neighbors=10, n_components=2, metric="cosine", random_state=42)
embeddings_2d=reducer_2d.fit_transform(embeddings)

plt.figure(figsize=(8,6))
plt.scatter(embeddings_2d[:,0],embeddings_2d[:,1],c=labels,cmap="tab10")
plt.title("UMAP + HDBSCAN Resume Clusters")
plt.show()

## IGNORE
# 2D visualization (for plotting only)

reducer_2d = umap.UMAP(
    n_neighbors=10,
    n_components=2,
    metric="cosine",
    random_state=42
)

embeddings_2d = reducer_2d.fit_transform(embeddings)

# 🔥 CLUSTER AGAIN IN 2D (ONLY FOR VISUALS)
clusterer_2d = hdbscan.HDBSCAN(min_cluster_size=4, min_samples=3)
labels_2d = clusterer_2d.fit_predict(embeddings_2d)

plt.figure(figsize=(8,6))
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=labels_2d, cmap="tab10")
plt.title("UMAP + HDBSCAN (2D Visualization Only)")
plt.show()

