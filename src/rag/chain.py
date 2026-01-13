from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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

def build_rag_chain(llm, retriever, prompt, question):
    parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()   
    })

    parsar = StrOutputParser()

    main_chain = parallel_chain | prompt | llm | parsar

    return main_chain.invoke(question)
    