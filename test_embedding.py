from src.embeddings.chunk_embed import get_resume_embedding

file_path="data/extracted_text/Abhishek Gupta.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

embedding = get_resume_embedding(text)

print("Embedding shape:", embedding.shape)