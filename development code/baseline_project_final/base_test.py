from transformers import T5ForConditionalGeneration, T5Tokenizer
#import preprocess
import baseline_preprocess
import re
import baseline_main
import baseline_train_multiple
import torch

def fetch_and_preprocess_ebook(ebook_id):
    print("*** Preprocess test book ***")
    text = baseline_main.fetch_gutenberg_ebook(ebook_id)
    start_match = re.search(r'\*{3}\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*{3}', text, re.DOTALL)
    end_match = re.search(r'\*{3}\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*{3}', text, re.DOTALL)
    combined_part_chapters = {}

    if start_match and end_match:
        start = start_match.end()  # Get the position after the start marker
        end = end_match.start()    # Get the position before the end marker
        extracted_text = text[start:end].strip()  # Extract the main content
        extracted_text = baseline_preprocess.remove_contents_section(extracted_text)

        # Preprocess the text
        part_chapters = baseline_preprocess.extract_parts_and_chapters(extracted_text)

        for part, (titles, contents) in part_chapters.items():
            if part not in combined_part_chapters:
                combined_part_chapters[part] = ([], [])
            combined_part_chapters[part][0].extend(titles)
            combined_part_chapters[part][1].extend(contents)

        print(f"*** Processed eBook ID {ebook_id}")
        print("--------------------------------")

        processed_data = baseline_preprocess.paragraph_processing(combined_part_chapters)
    else:
        print(f"Start or end markers not found for ebook ID {ebook_id}!")
        processed_data = None

    print("Length of processed data:", len(processed_data)) #[p1..pn][p1..pn]
    # Extract paragraphs from the nested list
    paragraphs = baseline_preprocess.flatten_paragraphs(processed_data)
    print("Length of paragraphs:", len(paragraphs))
    cleaned_paragraphs = [paragraph.replace("\r\n", " ") for paragraph in paragraphs]
    #print(cleaned_paragraphs[:3])
    baseline_preprocess.save_preprocess_paragraph(cleaned_paragraphs)

    return cleaned_paragraphs


# Function to generate summaries for each paragraph
def generate_paragraph_summaries(paragraphs, model, tokenizer):
    print("*** Start Generate Summary ***")
    paragraph_summaries = []
    for i, paragraph in enumerate(paragraphs):
        #print(f"Original paragraph {i+1}: {paragraph}...")  # Print the first 100 characters of the paragraph
        input_ids = tokenizer.encode(f"summarize: {paragraph}",
                                    return_tensors="pt",
                                    max_length=512, 
                                    truncation=True)
        #print(f"Input IDs for paragraph {i+1}: {input_ids}")
        outputs = model.generate(input_ids, 
                                max_length=512,
                                num_beams=2, 
                                length_penalty=2.0,
                                early_stopping=True)
        generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated summary {i+1}: {generated_summary}")
        paragraph_summaries.append(generated_summary)

    return paragraph_summaries

if __name__ == "__main__":
    ebook_id = 244
    test_data = fetch_and_preprocess_ebook(ebook_id)
    test_data = test_data[1:3]  # Limit to 3 paragraphs for testing
    if test_data:
        # Load model and tokenizer
        model_name = 't5-small'
	model, tokenizer = T5ForConditionalGeneration.from_pretrained(model_name), T5Tokenizer.from_pretrained(model_name)
        #model, tokenizer = baseline_train_multiple.load_model_and_tokenizer(model_name)
        
        # Generate summaries for each paragraph
        summaries = generate_paragraph_summaries(test_data, model, tokenizer)
        baseline_preprocess.save_summary(summaries)
        print("*** Summary done ***")
    else:
        print("No processed data available for generating summaries.")