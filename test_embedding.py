from src.preprocessing.clean_text import clean_text
from src.extraction.extract_text import extract_pdf_text
from src.embeddings.chunk_embed import get_resume_embedding

pdf_path="data/raw/Abhishek Gupta.pdf"

text = extract_pdf_text(pdf_path)
cleaned = clean_text(text)

embedding = get_resume_embedding(cleaned)

print("Embedding shape:", embedding.shape)