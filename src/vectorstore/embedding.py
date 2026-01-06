from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

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

def retrieve_similar_documents(vectorstore, query, k=5):
    """
    Retrieves similar documents from the vector store based on the given query.

    Args:
        vectorstore (FAISS): The FAISS vector store.
        query (str): The query string to search for similar documents.
        k (int): The number of similar documents to retrieve.
    Returns:
        list: A list of similar documents.
    """
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
    return retriever.get_relevant_documents(query)