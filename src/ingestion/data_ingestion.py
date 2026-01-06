from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled # fetch YouTube video transcripts using video-ID
from langchain_text_splitters import RecursiveCharacterTextSplitter # split text into smaller chunks

def fetch_youtube_transcript(video_id):
    """
    Fetches the transcript for a given YouTube video ID.

    Args:
        video_id (str): The YouTube video ID.
    Returns:
        str: The transcript text.
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=["en"])
        transcript_text = " ".join(chunk.text for chunk in transcript_list)
        return transcript_text
    except TranscriptsDisabled:
        return "Transcripts are disabled for this video."
    except Exception as e:
        return f"An error occurred: {str(e)}"
    

def create_chunks(chunk_size, chunk_overlap, text):
    """
    Splits the given text into smaller chunks.

    Args:
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.
        text (str): The text to be split.
    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.create_documents([text])
    return chunks