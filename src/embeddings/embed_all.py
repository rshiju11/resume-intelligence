"""
embed_all.py

- Embeds all resumes inside data/extracted_text
- Saves embedding matrix and filenames
"""

import os
import numpy as np
from src.embeddings.chunk_embed import get_resume_embedding
from sklearn.preprocessing import normalize

RAW_DIR ="data/extracted_text"
OUT_DIR="results/embeddings"

os.makedirs(OUT_DIR, exist_ok=True)

embeddings= []
filenames = []

for file in os.listdir(RAW_DIR):
    if file.lower().endswith(".txt"):
        path = os.path.join(RAW_DIR, file)
        print("Processing:", file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        

        emb= get_resume_embedding(text)

        if emb is not None:
            embeddings.append(emb)
            filenames.append(file)

# convert embeddings to matrix
embedding_matrix = np.vstack(embeddings)
embedding_matrix = normalize(embedding_matrix)

np.save(os.path.join(OUT_DIR, "resume_embeddings.npy"), embedding_matrix)

with open(os.path.join(OUT_DIR, "filenames.txt"), "w") as f:
    for name in filenames:
        f.write(name + "\n")

print("!! completed embedding resumes !!")
print("Embedding matrix:", embedding_matrix.shape)
