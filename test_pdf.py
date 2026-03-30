from src.preprocessing.split_sections import split_sections
from src.preprocessing.clean_text import clean_text
from src.extraction.extract_text import extract_pdf_text

pdf_path = "data/raw/Abhishek Gupta.pdf"

text = extract_pdf_text(pdf_path)
cleaned = clean_text(text)
sections = split_sections(cleaned)

for key, value in sections.items():
    print(f"\n{key.upper()}:")
    print(value)
    


