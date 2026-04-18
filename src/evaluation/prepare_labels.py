"""
prepare_labels.py

- Reads resume_groups.xlsx (Excel file)
- Matches resume filenames with embedding filenames
- Handles messy filename differences (pdf vs txt, spacing, symbols)
- Includes manual fixes for mismatched filenames
- Converts group labels into numeric labels
- Saves true_labels.npy for evaluation

Output:
data/processed/true_labels.npy
"""

import os
import numpy as np
import pandas as pd

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

EXCEL_PATH = os.path.join(BASE_DIR, "data", "resume_groups.xlsx")
FILENAMES_PATH = os.path.join(BASE_DIR, "results", "embeddings", "filenames.txt")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "true_labels.npy")


# -------------------------------
# HELPER: NORMALIZE FILENAMES
# -------------------------------
def normalize_name(name):
    name = name.lower()
    name = name.replace(".pdf", "").replace(".txt", "")
    name = name.replace("_", " ").replace("-", " ")
    name = "".join(c for c in name if c.isalnum() or c == " ")
    name = " ".join(name.split())
    return name


# -------------------------------
# LOAD EXCEL
# -------------------------------
df = pd.read_excel(EXCEL_PATH)

# clean column names (VERY IMPORTANT FIX)
df.columns = df.columns.str.strip().str.lower()

file_col = "resume"
label_col = "group"

# remove rows with missing labels
df = df.dropna(subset=[label_col])

# normalize filenames
df["clean_name"] = df[file_col].apply(normalize_name)


# -------------------------------
# LOAD EMBEDDING FILENAMES
# -------------------------------
with open(FILENAMES_PATH, "r") as f:
    embedding_filenames = [line.strip() for line in f.readlines()]

embedding_clean = [normalize_name(f) for f in embedding_filenames]


# -------------------------------
# MANUAL FIXES (EDGE CASES)
# -------------------------------
manual_map = {
    normalize_name("Hugh Barnaby _ ASU Search.txt"):
    normalize_name("Hugh Barnaby _ Ball State University.pdf")
}


# -------------------------------
# CREATE LABEL MAPPING
# -------------------------------
label_map = {label: idx for idx, label in enumerate(df[label_col].unique())}

print("Label Mapping:")
for k, v in label_map.items():
    print(f"{v} → {k}")


# -------------------------------
# MATCH FILENAMES
# -------------------------------
true_labels = []
missing = []

for orig_name, clean_name in zip(embedding_filenames, embedding_clean):

    # apply manual mapping if needed
    if clean_name in manual_map:
        target_name = manual_map[clean_name]
        row = df[df["clean_name"] == target_name]
    else:
        row = df[df["clean_name"] == clean_name]

    if len(row) == 0:
        missing.append(orig_name)
        true_labels.append(-1)
    else:
        label = row[label_col].values[0]
        true_labels.append(label_map[label])


true_labels = np.array(true_labels)


# -------------------------------
# SAVE OUTPUT
# -------------------------------
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
np.save(OUT_PATH, true_labels)

print("\nSaved true_labels.npy")
print("Shape:", true_labels.shape)


# -------------------------------
# DEBUG OUTPUT
# -------------------------------
if missing:
    print("\n⚠️ Missing matches:")
    for m in missing:
        print("-", m)
else:
    print("\nAll files matched successfully!")