import re
import os
import json
from datasets import Dataset

def remove_contents_section(text):
    #patterns to identify the "Contents" section, "PART", and "CHAPTER"
    contents_pattern = re.compile(r'\bContents\b', re.IGNORECASE)
    part_pattern = re.compile(r'\bPART\s+[IVXLCDM]+\b', re.IGNORECASE)
    chapter_pattern = re.compile(r'\bCHAPTER\b', re.IGNORECASE)
    
    # Search for the start of the "Contents" section
    contents_match = contents_pattern.search(text)
    
    if contents_match:
        # Find where the "Contents" section ends and start scanning for empty lines
        contents_end_index = contents_match.end()
        remaining_text = text[contents_end_index:]
        
        # Split remaining text into lines
        lines = remaining_text.splitlines()
        
        # Find the first non-empty line after multiple empty lines
        i = 0
        while i < len(lines) and not lines[i].strip():
            i += 1
        
        # Skip over up to 3 empty lines (if there are more, stop)
        empty_lines_count = 0
        while i < len(lines) and empty_lines_count <= 3:
            if not lines[i].strip():
                empty_lines_count += 1
            else:
                empty_lines_count = 0
            i += 1
        
        # Find the actual start of the main content by searching for the first part or chapter
        main_content_start_index = contents_end_index + sum(len(line) + 1 for line in lines[:i])
        main_content_start_match = part_pattern.search(text[main_content_start_index:])
        
        if not main_content_start_match:
            main_content_start_match = chapter_pattern.search(text[main_content_start_index:])
        
        if main_content_start_match:
            main_content_start_index = main_content_start_index + main_content_start_match.start()
            main_content = text[main_content_start_index:]
        else:
            # If no part or chapter start is found, use the computed index
            main_content = text[main_content_start_index:]
    else:
        # If no "Contents" section is found, look for the first part or chapter directly
        part_match = part_pattern.search(text)
        chapter_match = chapter_pattern.search(text)
        
        if part_match and (not chapter_match or part_match.start() < chapter_match.start()):
            main_content_start_index = part_match.start()
        elif chapter_match:
            main_content_start_index = chapter_match.start()
        else:
            main_content_start_index = 0
        
        main_content = text[main_content_start_index:]
    
    print("***cotent section has been removed***")
    return main_content.strip()


def extract_parts_and_chapters(text):
    print("***Extract Part and Chapter***")
    # Regex pattern to identify parts
    part_pattern = re.compile(r'(\bPart\s+\d+\b|\bPART\s+[IVXLCDM]+\b)', re.IGNORECASE)
    
    # Check for part matches
    part_matches = part_pattern.findall(text)

    if part_matches:
        print("***Part and Chapter Wise***")
        # If parts are found, process accordingly
        parts = part_pattern.split(text)
        part_chapters = {}

        for i in range(1, len(parts), 2):  # Skip the part title
            part_title = parts[i].strip()
            #print("***part_title:",part_title)
            part_content = parts[i + 1].strip() if (i + 1) < len(parts) else ""
            # Extract chapters for this part
            chapter_titles, chapter_contents = extract_chapters(part_content)
            part_chapters[part_title] = (chapter_titles, chapter_contents) 
       
        return part_chapters
    
    else:
        print("***Entire BOOK and Chapter wise***")
        # If no parts are found, treat the entire text as a single part
        chapter_titles, chapter_contents = extract_chapters(text)
        return {'Entire Book': (chapter_titles, chapter_contents)}
    
def extract_chapters(text):
    #part_pattern = re.compile(r'(Part \d+)', re.IGNORECASE)
    # Use regex to identify chapters
    chapter_pattern = re.compile(r'(\bChapter \d+\b|\bCHAPTER \w+\b)', re.IGNORECASE)
    
    # Split the text by chapter markers
    chapters = chapter_pattern.split(text)
    
    chapter_titles = []
    chapter_contents = []
    
    # Iterate through the split text
    for i in range(1, len(chapters), 2):  # Skip titles; start from the first chapter
        chapter_title = chapters[i].strip()
        chapter_content = chapters[i + 1].strip() if (i + 1) < len(chapters) else ""
        chapter_titles.append(chapter_title)
        chapter_contents.append(chapter_content)
    #print("chapter_titles:",chapter_titles )
    #print("chapter_content:",chapter_contents[0])#first chapter content
    return chapter_titles, chapter_contents
    
def extract_paragraphs(chapter_content):
    print("***extract paragraph***")
    # Split by two or more newlines or by other paragraph separators
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', chapter_content) if p.strip()]
    #print("paragraphs length:",len(paragraphs))
    #print("paragraph_content:",paragraphs[-1])#last paragraph
    return paragraphs

def flatten_paragraphs(preprocess_data):
    paragraphs = [para for chapter in preprocess_data for para in chapter]
    print("***Flatten Paragraph***")
    return paragraphs

def paragraph_processing(part_chapters):
    print("***paragraph cleaning***")
    paragraphs_data=[]
    chapters_data = {}
    for part, (titles, contents) in part_chapters.items():
        #print("***part:", part)
        chapters_data[part] = {}
        for title, chapter_content in zip(titles, contents):
            #print("***Chapter_title:", title)
            cleaned_content = clean_unicode_characters(chapter_content)
            paragraphs=extract_paragraphs(cleaned_content)
            paragraphs = [clean_unicode_characters(paragraph) for paragraph in paragraphs]
            #for i, paragraph in enumerate(paragraphs):
                #print(f"\n Paragraph {i + 1}:Lenght {len(paragraph)}\n")  
            chapters_data[part][title.replace('_', ' ')] = paragraphs
            paragraphs_data.append(paragraphs)
    print("***paragraph cleaning done***")
    return paragraphs_data
   
def clean_unicode_characters(text):
    # Define Unicode characters to replace
    replacements = {
        r'\u201c': '"',  # Left double quotation mark
        r'\u201d': '"',  # Right double quotation mark
        r'\u2018': "'",  # Left single quotation mark
        r'\u2019': "'",  # Right single quotation mark
        r'\u2022': '*',  # Bullet point
        r'\u2014' : '-'
        # Add other replacements if needed
    }
    # Replace characters
    for unicode_seq, char in replacements.items():
        text = re.sub(unicode_seq, char, text)
        #text=text.replace('\r\n', ' ')#\r\n line breaks with a space
    
    return text


def save_preprocess(preprocess_data):
    os.makedirs("preprocessed_data", exist_ok=True)
    preprocess_data = json.dumps(preprocess_data, indent=4)
    file_path = os.path.join("preprocessed_data", "preprocessed.json")

    # Write JSON string to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(preprocess_data)
    
    print(f"Data successfully saved to {file_path}")


def save_summary(preprocess_data):
    os.makedirs("generated_summary", exist_ok=True)
    preprocess_data = json.dumps(preprocess_data, indent=4)
    file_path = os.path.join("generated_summary", "summary.json")

    # Write JSON string to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(preprocess_data)
    
    print(f"Data successfully saved to {file_path}")



def save_preprocess_paragraph(paragraph):
    os.makedirs("paragraph_data", exist_ok=True)
    paragraph = json.dumps(paragraph, indent=4)
    file_path = os.path.join("paragraph_data", "paragraph.json")

    # Write JSON string to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(paragraph)
    
    print(f"Data successfully saved to {file_path}")




     