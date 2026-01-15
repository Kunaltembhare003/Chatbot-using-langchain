from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import  ChatOpenAI
from langchain_core.prompts import PromptTemplate

from config import OPENAI_API_KEY


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
        model="gpt-4o-mini",
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )

def format_docs(retrived_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrived_docs)
    return context_text

def build_rag_chain(llm, context, prompt, question):
    """
    Build and execute RAG chain with raw HNSW index.
    
    Args:
        llm: Language model instance
        context: Retrieved context string (from HNSW search results)
        prompt: Prompt template
        question: User question
    """
    # Format the prompt with context and question
    formatted_prompt = prompt.format(context=context, question=question)
    
    # Get response from LLM
    response = llm.invoke(formatted_prompt)
    
    return response.content
    