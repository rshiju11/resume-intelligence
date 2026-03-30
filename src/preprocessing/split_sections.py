#split_sections.py

"""
I have taken full extracted resume text here and split it 
into sections like : skills, experience and education

returns: section(dict) : a dictionary containing extracted text from each section
"""

def split_sections(text):
    # dictionary to store extracted sections from resume
    sections = {"skills":"", "experience":"", "education":""}
    
    # mapping of section names to all possible keywords
    # that may appear in resumes
    section_map = { "skills":["skills","technical skills"],
                   "experience": ["experience", "work experience", "professional experience"],
                   "education": ["education","academic","qualification"]}
    
    lines = text.split("\n")
    current_section = None

    for line in lines:
        if len(line.strip())==0:
            continue

        line_lower = line.lower()
        matched = False

        #Check if current line is a section heading
        for key, keywords in section_map.items():
            # checking if any keyword appears in the line
            # update current section when match found
            
            for k in keywords:
                if k in line_lower:
                    current_section= key
                    matched= True
                    break

            # if not a heading, add to current section
        if not matched and current_section:
            sections[current_section] += line+" "
            
    return sections


