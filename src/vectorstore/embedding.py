from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import OPENAI_API_KEY

def create_vectorstore(documents):
    """
    Creates a FAISS vector store from the given documents using OpenAI embeddings.

    Args:
        documents (list): A list of documents to be embedded and stored.
    Returns:
        FAISS: The created FAISS vector store.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def get_embeddings():
    """Returns embeddings object for loading vectorstore."""
    return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)