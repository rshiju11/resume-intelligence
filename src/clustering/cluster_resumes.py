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
SAVE_PATH = os.path.join(BASE_DIR, "results","clusters", "cluster_labels.npy")

embeddings = np.load(EMB_PATH)

print("embedding size:",embeddings.shape)

#PCA reduction

pca = PCA(n_components=50)
embeddings = pca.fit_transform(embeddings)

print("after PCA:",embeddings.shape)

#  UMAP reduction

reducer = umap.UMAP(n_neighbors=5, n_components=20, min_dist=0.0,metric="cosine", random_state=42)
reduced_embeddings = reducer.fit_transform(embeddings)
print("reduced embedding size:", reduced_embeddings.shape)

# HDBSCAN cluster
clusterer = hdbscan.HDBSCAN(min_cluster_size=6, metric="euclidean", min_samples=2, cluster_selection_method='leaf')
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
reducer_2d = umap.UMAP(n_neighbors=5, n_components=2, min_dist=0.1 , metric="cosine", random_state=42)
embeddings_2d=reducer_2d.fit_transform(embeddings)

plt.figure(figsize=(8,6))
plt.scatter(embeddings_2d[:,0],embeddings_2d[:,1],c=labels,cmap="tab10")
plt.title("UMAP + HDBSCAN Resume Clusters")
plt.show()

