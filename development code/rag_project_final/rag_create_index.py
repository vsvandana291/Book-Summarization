import torch
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')#all-mpnet-base-v2 1


def embed_and_index_paragraphs(paragraphs, batch_size=32):
    print("***create embed and index")
    all_embeddings = []
    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i+batch_size]
        batch_embeddings = sbert_model.encode(batch, convert_to_numpy=True)
        all_embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(all_embeddings)
    
    # Create FAISS index
    d = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)  # L2 distance index
    index.add(embeddings)
    
    # Save the FAISS index
    faiss.write_index(index, "rag_index.faiss")
    return index



