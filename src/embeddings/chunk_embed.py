"""
chunk embed.py

This module :
- splits resume text into semantic chunks
- generates embeddings for each chunk (BGE)
- aggregates them into single resume embedding
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("BAAI/bge-large-en-v1.5")

INSTRUCTION = "Represent this resume for clustering based on research focus, technical expertise, and career domain."

ANCHORS = [
    "machine learning, artificial intelligence, deep learning, neural networks",
    "data science, statistics, data analysis, predictive modeling",
    "software engineering, backend development, distributed systems",
    "computer vision, NLP, natural language processing",
    "academic research, publications, conferences, PhD research"
]



def chunk_text(text, chunk_size=500):
    """
    This splits text into word based chunks
    Returns list of text chunks
    """

    words  = text.split()
    chunks =[]

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    
    return chunks

def embed_chunks(chunks):
    """
    converts list of chunks into embeddings.
    returns array of embeddings.
    """
    if not chunks:
        return None
    
    formatted = [INSTRUCTION + " " + chunk for chunk in chunks]
    embeddings = model.encode(formatted, normalize_embeddings=True)

    return np.array(embeddings)

def filter_relevant_chunks(chunks, embeddings, ratio = 0.7):
    """
    Filters chunks based on semantic similarity to anchor,
    keeps top 70% of the chunks
    """

    if embeddings is None:
        return None, None
    
    # embed anchor, compute cos. sim. between chunk and anchor
    anchor_embeddings = model.encode([INSTRUCTION + " " + a for a in ANCHORS], normalize_embeddings=True)
    
    sims = cosine_similarity(embeddings, anchor_embeddings)
    max_sims = sims.max(axis=1)

    # keeping top k (0.7) chunks
    k = max(1, int(len(chunks)*ratio))
    top_indices = np.argsort(-max_sims)[:k]

    return embeddings[top_indices], max_sims[top_indices]


def aggregate_embeddings(embeddings, weights=None):
    """
    chunk embeddings are aggregated into a single resume vector
    returns final aggregated embedding.
    """
    if embeddings is None:
        return None
    
    if weights is not None:
        if np.sum(weights) == 0:
            weights = None
    
    if weights is None:
        final = np.mean(embeddings, axis=0)
    else:
        weights = np.exp(weights) / np.sum(np.exp(weights))
        final = np.sum(embeddings * weights[:, None], axis=0)
    
    # normalize final embeddings for clustering
    final = final/np.linalg.norm(final)
    
    return final

def get_resume_embedding(text):
    """
    text -> chunks -> embeddings -> filter -> average
    """
    chunks = chunk_text(text)
    chunk_embs= embed_chunks(chunks)
    filtered_embs, weights = filter_relevant_chunks(chunks,chunk_embs,ratio=0.7)
    final_embedding = aggregate_embeddings(filtered_embs, weights)
    return final_embedding



