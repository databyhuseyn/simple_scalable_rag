from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__  as core_version
from langgraph import version as lg_version
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import os

from app.document_loaders import pdf_loader
from app.chunker import chunk_documents
from app.vector_db import store_vectors


def main():

    pdfs = pdf_loader()
    chunks = chunk_documents(pdfs)
    vectors = store_vectors(chunks)

    print('Success...')

    query = "What are the main causes of engine failure?"
    res = vectors.similarity_search_with_score(
        query,
        k=1
    )

    for doc, score in res:
        print("Score:", score)
        print("Text:", doc.page_content[:50])
        print("Source:", doc.metadata)
        print('-' * 50)



    # retriever = vectors.as_retriever(
    # search_kwargs={
    #     "k": 5
    # })

    # docs = retriever.invoke(
    #     "What are the main causes of engine failure?"
    # )

    # docs

    # print(docs)


if __name__ == "__main__":
    main()
