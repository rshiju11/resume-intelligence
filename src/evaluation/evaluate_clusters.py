"""
evaluate_clusters.py

evaluates cluster quality using:
- Noise ratio
- Cluster size distribution
- Intra-cluster similarity
- Inter-cluster similarity
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import umap
import os


# load data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
EMB_PATH = os.path.join(BASE_DIR, "results", "embeddings", "resume_embeddings.npy")
LABEL_PATH = os.path.join(BASE_DIR, "results", "clusters", "cluster_labels.npy")

embeddings = np.load(EMB_PATH)
labels = np.load(LABEL_PATH)

print("Loaded embeddings:", embeddings.shape)
print("Loaded labels:", labels.shape)

# PCA (for similarity metrics)
pca = PCA(n_components=50)
embeddings_pca =pca.fit_transform(embeddings)
embeddings_pca = normalize(embeddings_pca)

# UMAP space (for clustering evaluation: DB index)
reducer = umap.UMAP(n_neighbors=5,n_components=20,min_dist=0.0,metric="cosine",random_state=42)
embeddings_umap = reducer.fit_transform(embeddings_pca)

print("Transformed embeddings for evaluation:")
print("PCA shape ", embeddings_pca.shape)
print("UMAP shape ", embeddings_umap.shape)

# noise ratio
num_noise = np.sum(labels == -1)
noise_ratio =num_noise/len(labels)

print("\nNoise points:",num_noise)
print("Noise ratio:",round(noise_ratio, 3))


# cluster sizes

cluster_ids = sorted([c for c in set(labels) if c != -1])
print("\nCluster size distribution:")

for cid in cluster_ids:
    size = np.sum(labels==cid)
    print(f"Cluster {cid}: {size} resumes")


# intra cluster similarity

print("\nIntra-cluster similarity:")
intra_values = []

for cid in cluster_ids:
    cluster_points = embeddings_pca[labels == cid]
    if len(cluster_points) > 1:
        sim_matrix = cosine_similarity(cluster_points)
        np.fill_diagonal(sim_matrix, 0)

        n = sim_matrix.shape[0]
        mean_sim = np.sum(sim_matrix) / (n*n - n)

        intra_values.append(mean_sim)
        print(f"Cluster {cid}: {round(mean_sim,3)}")

# inter cluster similarity 
print("\nInter-cluster centroid similarity:")
centroids =[]
for cid in cluster_ids:
    centroid =np.mean(embeddings_pca[labels == cid],axis=0)
    centroid =centroid/np.linalg.norm(centroid)
    centroids.append(centroid)

centroids =np.array(centroids)

if len(centroids)>1:
    centroid_sim =cosine_similarity(centroids)
    print(np.round(centroid_sim,3))

    mean_intra = np.mean(intra_values) if intra_values else 0

    print("\nMean intra-cluster similarity:",round(mean_intra,3))

    mean_inter = np.mean(centroid_sim[np.triu_indices_from(centroid_sim, k=1)])
    print("Mean inter-cluster similarity:",round(mean_inter,3))
    print("Separation gap (intra - inter):",round(mean_intra-mean_inter,3))

"""
Davies Bouldin index

Interpretation:
- lower is better
< 1.0  : strong clustering
1 to 2  : moderate structure
> 2    : weak separation
"""


# DB index cannot handle noise (-1), so removing noise points
valid_mask= labels != -1

if len(set(labels[valid_mask]))> 1:
    db_score =davies_bouldin_score(embeddings_umap[valid_mask],labels[valid_mask])
    print("\nDavies-Bouldin Index:",round(db_score,3))
    print()
else:
    print("\nDavies-Bouldin cannot be computed (only one cluster).")

if len(set(labels[valid_mask])) > 1:
    sil_score = silhouette_score(
        embeddings_umap[valid_mask],
        labels[valid_mask]
    )
    print("Silhouette Score:", round(sil_score, 3))

