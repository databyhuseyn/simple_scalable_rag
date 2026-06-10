from dotenv import load_dotenv
import os
load_dotenv()

# core dependencies
from langchain_core import __version__  as core_version
from langgraph import version as lg_version
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# local dependencies
from app.document_loaders import pdf_loader
from app.chunker import chunk_documents
from app.vector_db import store_vectors
from app.formatter import format_docs


def main():

    pdfs = pdf_loader()
    chunks = chunk_documents(pdfs)
    vectors = store_vectors(chunks)

    print('Success...')


    retriever = vectors.as_retriever(search_type='similarity', search_kwargs={'k':5})

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant which answers questions based on context given below. Answer in conscise manner. 
    If you don't know the answer or if the relevant information is not in context, answer with "I don't know".

    Context:
    {context}

    Question:
    {question}
                                              
    Answer:""")

    llm = ChatOpenAI(
        model = os.getenv("MODEL_NAME"), 
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url = os.getenv("BASE_URL"),
        streaming=True
    )

    rag_chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        'Game of Thrones serialında Eddard Stark niyə öldürülmüşdü? Ətraflı izah et'
    ]

    for q in questions:
        answer = rag_chain.invoke(q)
        print(f'Q: {q}')
        print(f'A: {answer}')


    # query = "What are the main causes of engine failure?"
    # res = vectors.similarity_search_with_score(
    #     query,
    #     k=1
    # )

    # for doc, score in res:
    #     print("Score:", score)
    #     print("Text:", doc.page_content[:50])
    #     print("Source:", doc.metadata)
    #     print('-' * 50)



if __name__ == "__main__":
    main()
