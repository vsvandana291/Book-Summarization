import requests
#import preprocess
import rag_preprocess
import re
import rag_create_index
import os
import json


def fetch_gutenberg_ebook(ebook_id):
    url = f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-0.txt"
    response = requests.get(url)

    if response.status_code == 200:
        response.encoding = 'utf-8'
        return response.text
    else:
        raise Exception(f"Failed to fetch the eBook: {response.status_code}")


def main():
    ebook_ids = [244, 394, 345]  # Add multiple ebook IDs
    combined_part_chapters = {}

    try:
        for ebook_id in ebook_ids:
            print(f"***Fetching eBook ID {ebook_id}")
            text = fetch_gutenberg_ebook(ebook_id)
            print(f"***Extracted text length for eBook ID {ebook_id}: {len(text)}")
            
            start_match = re.search(r'\*{3} START OF THE PROJECT GUTENBERG EBOOK.*?\*{3}', text)
            end_match = re.search(r'\*{3} END OF THE PROJECT GUTENBERG EBOOK.*?\*{3}', text)

            if start_match and end_match:
                start = start_match.end()  # Get the position after the start marker
                end = end_match.start()    # Get the position before the end marker
                extracted_text = text[start:end].strip()  # Extract the main content
                #print(f"Main content length for eBook ID {ebook_id}: {len(extracted_text)}")
                
                extracted_text = rag_preprocess.remove_contents_section(extracted_text)
                #print(f"Content length after removing contents section for eBook ID {ebook_id}: {len(extracted_text)}")

                part_chapters = rag_preprocess.extract_parts_and_chapters(extracted_text)
                #print(f"Extracted parts and chapters for eBook ID {ebook_id}: {list(part_chapters.keys())}")
                #print(f"Part chapters: {part_chapters}")

                for part, (titles, contents) in part_chapters.items():
                    if part not in combined_part_chapters:
                        combined_part_chapters[part] = ([], [])
                    #print(f"Processing eBook ID {ebook_id} - Part: {part}")
                    combined_part_chapters[part][0].extend(titles)
                    combined_part_chapters[part][1].extend(contents)
                    #print(f"Combined titles for {part}: {combined_part_chapters[part][0][:3]}")
                    #print(f"Combined contents (first 100 chars) for {part}: {[content[:100] for content in combined_part_chapters[part][1][:3]]}")

                print(f"***Processed eBook ID {ebook_id}")
                print("--------------------------------")

            else:
                print(f"Start or end markers not found for ebook ID {ebook_id}!")


        preprocess_data = rag_preprocess.paragraph_processing(combined_part_chapters)
        #rag_preprocess.save_preprocess(preprocess_data)
        
        print("lenght preprocess_data:",len(preprocess_data))#[p1..pn][p1..pn]
        # Extract paragraphs from the nested list
        paragraphs = rag_preprocess.flatten_paragraphs(preprocess_data)
        print("lenght paragrah_data:", len(paragraphs))
        #print(paragraphs[-1])
        rag_preprocess.save_preprocess_paragraph(paragraphs)

        # Embed and index the combined paragraphs
        index = rag_create_index.embed_and_index_paragraphs(paragraphs)
        print("Indexing completed successfully for combined data.")

        cleaned_paragraphs = [paragraph.replace("\r\n", " ") for paragraph in paragraphs]
        with open("test_paragraphs.txt", "w") as f:
                print("############check###############")
                for para in cleaned_paragraphs:
                    if para.strip():
                        f.write(para +"\n\n")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()