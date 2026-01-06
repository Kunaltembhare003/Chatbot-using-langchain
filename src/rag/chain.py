from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from vectorstore.embedding import create_vectorstore, retrieve_similar_documents
from langchain_core.prompts import PromptTemplate
from config import OPENAI_API_KEY, LLM_MODEL

prompt = PromptTemplate(
    template="""
        You are a helpful AI assistant.
        Answer the question based on the provided context.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}
    """,
    input_variables=["context", "question"]
)

def build_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )

