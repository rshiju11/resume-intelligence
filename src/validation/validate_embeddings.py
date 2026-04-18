"""
validate_embeddings.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


embeddings = np.load("results/embeddings/resume_embeddings.npy")
print("\n\nembeddings size:", embeddings.shape)

"""
Normality check: 

- If morm ~ 0.9 to 1 : 
    embeddings are properly normalized, 
    cosine similarity will behave properly
    safe for clustering

- If norm varies 0.6 to 1.4:
    normalization not applied

- if norm 0 : 
    failed chunk, breaks in embedding

"""

norms = np.linalg.norm(embeddings, axis=1)

print("\nNormalization check:")
print("Min norm:",norms.min())
print("Max norm:",norms.max())
print("Mean norm:",norms.mean())


"""
Cosine similarity check: scores ~ 0.3 to 0.6 suggests useful semantic variation.
higher values may mean embedding collapse
lower values may mean weak structure 

Histogram: widespread means good semantic diversity
single peak near high similarity = too homogenous
flat random distrib. = weak str
"""

similarity_matrix = cosine_similarity(embeddings)

# remove diagonals
sim_values = similarity_matrix[np.triu_indices_from(similarity_matrix,k=1)]

print("\nSimilarity stats:")
print("Min:", sim_values.min())
print("Max:", sim_values.max())
print("Mean:", sim_values.mean())

plt.hist(sim_values, bins=50)
plt.title("Pairwise Cosine Similarity Distribution")
plt.show()

"""
Nearest neighbor check:  if semantically similar resumes are close in embedding space
"""

with open("results/embeddings/filenames.txt","r") as f:
    filenames =[line.strip() for line in f.readlines()]

for i in range(5):
    sims = cosine_similarity([embeddings[i]], embeddings)[0]
    nearest_idx = np.argsort(-sims)[1:6]        #top 5 neighbors

    print("\nResume:",filenames[i])
    print("Nearest neighbors:")
    for idx in nearest_idx:
        print("   ", filenames[idx], "| similarity:", round(sims[idx], 3))


"""
structure check: for a visible data spread 

- visible spread /clusters forming: structured embeddings, meaningful clusters
- single spread out blob: still clusterable
- tight dot: collapsed embeddings 

"""
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

plt.scatter(reduced[:,0],reduced[:,1])
plt.title("PCA Projection of Resume Embeddings")
plt.show()









