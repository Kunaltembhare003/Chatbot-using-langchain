import os
from src.ingestion.data_ingestion import fetch_youtube_transcript, create_chunks
from src.vectorstore.embedding import create_vectorstore,  get_embeddings
from src.rag.chain import prompt, build_llm, build_rag_chain
from langchain_community.vectorstores import FAISS

VECTORSTORE_PATH = "./vectorstore"


def load_or_create_vectorstore(video_id):
    """Load cached vectorstore or create new one."""
    vectorstore_file = os.path.join(VECTORSTORE_PATH, f"{video_id}_vectorstore")
    
    # If vectorstore exists, load it
    if os.path.exists(vectorstore_file):
        print(f"Loading cached vectorstore for {video_id}...")
        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(vectorstore_file, 
                                       embeddings,
                                       allow_dangerous_deserialization=True)
        return vectorstore
    
    # Otherwise, create new vectorstore
    print("Creating new vectorstore...")
    transcript = fetch_youtube_transcript(video_id)
    
    if transcript and not transcript.startswith("An error occurred"):
        chunks = create_chunks(chunk_size=1000, chunk_overlap=200, text=transcript)
        print(f"Created {len(chunks)} chunks")
        
        vectorstore = create_vectorstore(chunks)
        
        # Save vectorstore
        os.makedirs(VECTORSTORE_PATH, exist_ok=True)
        vectorstore.save_local(vectorstore_file)
        print(f"Vectorstore saved to {vectorstore_file}")
        
        return vectorstore
    else:
        print(f"Failed to fetch transcript: {transcript}")
        return None

video_id = input("Enter YouTube video ID:") or "Gfr50f6ZBvo"
vectorstore = load_or_create_vectorstore(video_id)

if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    question = input("Ask Question: ") or "Summarize the video in brief."
    llm = build_llm()
      

    response = build_rag_chain(llm, retriever, prompt, question)
    # Get response from LLM
    print(f"Answer: {response}")