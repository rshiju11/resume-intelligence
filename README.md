## Semantic Resume Clustering using Transformer Embeddings, UMAP and HDBSCAN

This project clusters technical resumes based on semantic similarity using an unsupervised learning pipeline. Resumes are converted into embeddings using a transformer model, followed by dimensionality reduction and clustering.

## Author
Rshijuta Pokharel
MS Data Science, Wright State University

## Dataset
1. Total resumes: 73
2. Format: PDF files
3. Location: data/raw/

## Pipeline Steps
1. Text Extraction: Extracts and cleans text from PDF resumes.
2. Embedding Generation: Generates semantic embeddings using BAAI (BGE model).
3. Dimensionality Reduction: Applies PCA and UMAP.
4. Clustering: Uses HDBSCAN to form clusters.
5. Evaluation: Uses internal and external metrics.

## Requirements
pip install -r requirements.txt

## How to Run
1. Extract Text: python src/preprocessing/extract_text.py
2. Generate Embeddings: python src/embeddings/embed_all.py
3. Clustering: python src/clustering/cluster_resumes.py
4. Validate Embeddings: python src/validation/validate_embeddings.py
5. Prepare Labels for evaluation: python src/evaluation/prepare_labels.py
6. Evaluate Clusters: python src/evaluation/evaluate_clusters.py
 
## Results
1. Embeddings: results/embeddings/resume_embeddings.npy
2. Cluster labels: results/clusters/
3. Evaluation results: Printed in terminal