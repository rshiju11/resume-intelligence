from src.preprocessing.split_sections import split_sections
from src.preprocessing.clean_text import clean_text

sample_text = """
Skills
Python, SQL, Machine Learning

Experience
Data Scientist at Google

Education
MS in Data Science

"""

cleaned =clean_text(sample_text)
sections =split_sections(cleaned)

print(sections)