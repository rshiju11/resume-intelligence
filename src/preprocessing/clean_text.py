#clean_text.py

"""
this fucntion cleans each line individually, 
preserves newline breaks AND
keep section detection working
"""

def clean_text(text):
    lines = text.split("\n")
    cleaned_lines = [" ".join(line.split()) for line in lines]
    return "\n ".join(cleaned_lines)