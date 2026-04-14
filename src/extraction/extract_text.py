# extract_text.py

import os
import re
import pdfplumber
import wordninja

RAW_DIR = "data/raw"
OUT_DIR = "data/extracted_text"


"""
this fucntion cleans each line individually, 
preserves newline breaks , skips empty lines,
keep section detection working and more:
"""

def clean_text(text):
    lines = text.split("\n")
    cleaned_lines = []

    for i, line in enumerate(lines):
        # normalize spacing
        line = " ".join(line.split())

        # skip empty lines
        if not line:
            continue

        # remove emails
        line = re.sub(r'\S+@\S+', ' ', line)

        # remove websites / URLs
        line = re.sub(r'(https?://\S+|www\.\S+|\S+\.(com|org|edu|net|io|ai|co))', ' ', line)

        # remove phone numbers
        line = re.sub(r'\+?\d[\d\-\(\) ]{8,}\d', ' ', line)

        # remove timestamps
        line = re.sub(r'\b\d{1,2}:\d{2}(\s?[ap]m)?\b', ' ', line)

        # remove years (2020, 1998)
        line = re.sub(r'\b(19|20)\d{2}\b', '', line)
        line = re.sub(
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}',
            ' ',
            line,
            flags=re.IGNORECASE
        )


        # camelCase
        line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)

        # fix missing space AFTER punctuation (but not before)
        line = re.sub(r'([.,])([A-Za-z])', r'\1 \2', line)

        # remove date formats 
        line = re.sub(r'\b\d{1,2}[/-]\d{4}\b', ' ', line)

        # remove unwanted keywords
        #keywords = ["from", "for","and","of"]
        #for kw in keywords:
        #    line = line.replace(kw, ' ')

        # remove names ONLY in first 2 lines

        if i < 2:
            line = re.sub(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', ' ', line)
        
        # fix merged words
        line = fix_merged_words(line)

        # clean extra spaces again
        line = " ".join(line.split())

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

# fix merged words (word segmentation)

def fix_merged_words(line):
    words = line.split()
    fixed_words = []

    for word in words:
        # skip structured tokens
        if "-" in word or "." in word:
            fixed_words.append(word)
            continue

        # only attempt split for long words
        if len(word) > 10:
            split_words = wordninja.split(word)

            # accept split only if meaningful
            if len(split_words) > 1:
                fixed_words.extend(split_words)
            else:
                fixed_words.append(word)
        else:
            fixed_words.append(word)

    return " ".join(fixed_words)



def extract_pdf_text(pdf_path):
    text=""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3)

                # fallback if spacing is bad or None
                if not page_text:
                    page_text = page.extract_text(layout=True)

                if page_text:
                    text += page_text.strip() + "\n"

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text 

def process_all_pdfs():
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(RAW_DIR,filename)
            txt_filename = filename.replace(".pdf",".txt")
            txt_path = os.path.join(OUT_DIR, txt_filename)
            text=extract_pdf_text(pdf_path)
            text=clean_text(text)
            # handle short/empty extractions
            if len(text.strip())<50:
                print(f"Warning: low text extracted from {filename}")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"Processed: {filename} --- to --- {txt_filename}")

if __name__ == "__main__":
    process_all_pdfs()