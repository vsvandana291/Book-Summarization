import requests
#import preprocess
import baseline_preprocess
import re
#import training_multiple
import baseline_train_multiple
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
    ebook_ids = [244, 394, 345] 
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
                
                extracted_text = baseline_preprocess.remove_contents_section(extracted_text)
                #print(f"Content length after removing contents section for eBook ID {ebook_id}: {len(extracted_text)}")

                part_chapters = baseline_preprocess.extract_parts_and_chapters(extracted_text)
                #print(f"Extracted parts and chapters for eBook ID {ebook_id}: {list(part_chapters.keys())}")
                #print(f"Part chapters: {part_chapters}")

                for part, (titles, contents) in part_chapters.items():
                    if part not in combined_part_chapters:
                        combined_part_chapters[part] = ([], [])
                    combined_part_chapters[part][0].extend(titles)
                    combined_part_chapters[part][1].extend(contents)
                    
                print(f"***Processed eBook ID {ebook_id}")
                print("--------------------------------")

            else:
                print(f"Start or end markers not found for ebook ID {ebook_id}!")
                
        print("############check###############")
        preprocess_data = baseline_preprocess.paragraph_processing(combined_part_chapters)
        #baseline_preprocess.save_preprocess(preprocess_data)

        print("lenght preprocess_data:",len(preprocess_data))
        # Extract paragraphs from the nested list
        paragraphs = baseline_preprocess.flatten_paragraphs(preprocess_data)
        print("lenght paragrah_data:", len(paragraphs))

        # preprocess dataset for t5 model:
        dataset = baseline_train_multiple.prepare_dataset(preprocess_data)
        
        model_name = 't5-small'
        model, tokenizer = baseline_train_multiple.train_chapter(dataset, model_name)
        
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()