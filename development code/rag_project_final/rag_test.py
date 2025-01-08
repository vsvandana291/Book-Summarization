import faiss
import rag_query_index
import faiss
import os
import gradio as gr

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


iface = gr.Interface(fn=rag_query_index.gradio_app, inputs="text", outputs="text", title="RAG Summarization App",
                     description="")

iface.launch(share=True)