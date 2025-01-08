import faiss
from sentence_transformers import SentenceTransformer
from transformers import  GPT2Tokenizer, GPT2LMHeadModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import pipeline

# Load BART model and tokenizer for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load FAISS index
index = faiss.read_index("rag_index.faiss")

with open("test_paragraphs.txt", "r") as f:
    paragraphs = f.read().splitlines()

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load GPT-2 model and tokenizer
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load T5 model and tokenizer for summarization
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    return index


def query_faiss_index(query, k=5):
    query_embedding = sbert_model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, k)
    retrieved_paragraphs = [paragraphs[i] for i in I[0]]
    return retrieved_paragraphs


def generate_response(query):
    retrieved_paragraphs = query_faiss_index(query)
    # Construct the context from the retrieved paragraphs
    context = " ".join(retrieved_paragraphs)
    
    # Generate the summary using BART
    #summary = summarizer(context, max_length=500, min_length=30, do_sample=False)[0]['summary_text']
    summary = summarizer(context, max_length=500, min_length=30, do_sample=False, num_beams=5,
                          length_penalty=2.0, early_stopping=True)[0]['summary_text']
    return summary

def gradio_app(query):
    response = generate_response(query)
    return response