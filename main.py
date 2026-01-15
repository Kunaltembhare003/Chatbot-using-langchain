import os
import json
import numpy as np
from src.ingestion.data_ingestion import fetch_youtube_transcript, create_chunks
from src.vectorstore.embedding import create_vectorstore, get_embeddings
from src.rag.chain import prompt, build_llm, build_rag_chain
import faiss

VECTORSTORE_PATH = "./vectorstore"


def load_or_create_vectorstore(video_id):
    """Load cached vectorstore or create new one."""
    index_file = os.path.join(VECTORSTORE_PATH, f"{video_id}_HNSW.faiss")
    metadata_file = os.path.join(VECTORSTORE_PATH, f"{video_id}_metadata.json")
    
    # If vectorstore exists, load it
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        print(f"Loading cached vectorstore for {video_id}...")
        hnsw_index = faiss.read_index(index_file)
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return hnsw_index, metadata["chunk_texts"]
    
    # Otherwise, create new vectorstore
    print("Creating new vectorstore...")
    transcript = fetch_youtube_transcript(video_id)
    
    if transcript and not transcript.startswith("An error occurred"):
        chunks = create_chunks(chunk_size=1000, chunk_overlap=200, text=transcript)
        print(f"Created {len(chunks)} chunks")
        
        hnsw_index = create_vectorstore(chunks, M=5, ef_construction=10)
        
        # Extract chunk texts for metadata
        chunk_texts = [chunk.page_content if hasattr(chunk, 'page_content') else chunk for chunk in chunks]
        
        # Save vectorstore and metadata
        os.makedirs(VECTORSTORE_PATH, exist_ok=True)
        faiss.write_index(hnsw_index, index_file)
        
        metadata = {"chunk_texts": chunk_texts}
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        print(f"Vectorstore saved to {index_file}")
        print(f"Metadata saved to {metadata_file}")
        
        return hnsw_index, chunk_texts
    else:
        print(f"Failed to fetch transcript: {transcript}")
        return None, None

video_id = input("Enter YouTube video ID: ") or "Gfr50f6ZBvo"
result = load_or_create_vectorstore(video_id)

if result[0] is not None:
    vectorstore, chunk_texts = result
    question = input("Ask Question: ") or "Summarize the video in brief."
    query_embedding = get_embeddings().embed_query(question)
    query_array = np.array([query_embedding], dtype="float32")
    
    # Search in HNSW index
    distances, indices = vectorstore.search(query_array, k=3)
    
    # Retrieve relevant chunks
    retrieved_chunks = [chunk_texts[idx] for idx in indices[0]]
    context = "\n\n".join(retrieved_chunks)
    
    llm = build_llm()
    response = build_rag_chain(llm, context, prompt, question)
    
    print(f"\nAnswer: {response}")