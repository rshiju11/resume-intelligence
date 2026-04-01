import os
import re
import numpy as np
from collections import Counter

# paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TEXT_DIR = os.path.join(BASE_DIR, "data", "extracted_text")
LABEL_PATH = os.path.join(BASE_DIR, "results", "clusters", "cluster_labels.npy")

# load labels
labels = np.load(LABEL_PATH)

# your noise words 
NOISE_WORDS = {
   # "and", "for", "the", "with", "engineering", "from", "ieee"
}


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


# load documents
documents = []
filenames = sorted(os.listdir(TEXT_DIR))

for filename in filenames:
    if filename.endswith(".txt"):
        with open(os.path.join(TEXT_DIR, filename), "r", encoding="utf-8") as f:
            documents.append(f.read())


# group documents by cluster
clusters = {}
for i, label in enumerate(labels):
    if label == -1:
        continue
    clusters.setdefault(label, []).append((filenames[i], documents[i]))


# INTERPRETATION
for cid, docs in clusters.items():
    counter = Counter()

    for fname, doc in docs:
        words = preprocess(doc)

        filtered = [
            w for w in words
            if w not in NOISE_WORDS and len(w) > 2
        ]

        counter.update(filtered)

    print(f"\n{'='*50}")
    print(f"Cluster {cid} (size={len(docs)})")
    print("="*50)

    print("\nSample resumes:")
    for fname, _ in docs[:8]:
        print("-", fname)