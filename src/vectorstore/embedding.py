from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import OPENAI_API_KEY
import faiss
import numpy as np

def create_vectorstore(documents, M=5, ef_construction=10):
    """
    Creates a FAISS HNSW vector store from the given documents using OpenAI embeddings.

    Args:
        documents (list): A list of Document objects or strings to be embedded and stored.
        M (int): Number of bi-directional links for HNSW index. Default is 5.
        ef_construction (int): Construction parameter for HNSW index. Default is 10.
    Returns:
        faiss index: The created HNSW FAISS index.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    
    # Extract text from Document objects if needed
    doc_texts = [doc.page_content if hasattr(doc, 'page_content') else doc for doc in documents]
    
    # Generate embeddings (this uses your OpenAI API)
    chunk_embeddings = embeddings.embed_documents(doc_texts)
    # Convert to numpy array (required by FAISS)
    embedding_array = np.array(chunk_embeddings, dtype="float32")
    # NOTE: LangChain's FAISS wrapper does NOT support HNSW directly
    # Use raw 'faiss' module instead (not the 'FAISS' LangChain class)
    dimension = embedding_array.shape[1]
    # Create the index using raw faiss module
    hnsw_index = faiss.IndexHNSWFlat(dimension, M)
    hnsw_index.hnsw.ef = ef_construction
    # Add your actual embeddings
    hnsw_index.add(embedding_array)
    print(f"HNSW index created with {hnsw_index.ntotal} vectors")
    return hnsw_index

def get_embeddings():
    """Returns embeddings object for loading vectorstore."""
    return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

