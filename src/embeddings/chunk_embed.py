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

ANCHOR_TEXT = "Technical research expertise, specialization, and professional domain focus."



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
        return None
    
    # embed anchor, compute cos. sim. between chunk and anchor
    anchor_embedding = model.encode(INSTRUCTION + " " + ANCHOR_TEXT, normalize_embeddings=True)
    sims = cosine_similarity(embeddings, [anchor_embedding]).flatten()

    # keeping top k (0.7) chunks
    k = max(1, int(len(chunks)*ratio))
    top_indices = np.argsort(-sims)[:k]

    filtered_embeddings = embeddings[top_indices]

    return filtered_embeddings


def aggregate_embeddings(embeddings):
    """
    chunk embeddings are aggregated into a single resume vector
    returns final aggregated embedding.
    """
    if embeddings is None:
        return None
    
    final = np.mean(embeddings,axis=0)
    
    # normalize final embeddings for clustering
    final = final/np.linalg.norm(final)
    
    return final

def get_resume_embedding(text):
    """
    text -> chunks -> embeddings -> filter -> average
    """
    chunks = chunk_text(text)
    chunk_embs= embed_chunks(chunks)
    filtered_embs = filter_relevant_chunks(chunks,chunk_embs,ratio=0.7)
    final_embedding = aggregate_embeddings(filtered_embs)
    return final_embedding



