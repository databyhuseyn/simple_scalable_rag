# core dependencies
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# local dependencies

from app.formatter import format_docs
from app.retriever import get_retriever
from app.llm import get_llm
from app.reranker import rerank


def main():

    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant which answers questions based on context given below. Answer in conscise manner. 
    If you don't know the answer or if the relevant information is not in context, answer with "I don't know".

    Context:
    {context}

    Question:
    {question}
                                              
    Answer:""")

    llm = get_llm()

    
    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        'Game of Thrones serialında Eddard Stark niyə öldürülmüşdü? Ətraflı izah et'
    ]

    for q in questions:

        docs = retriever.invoke(q)
        docs = rerank(
            q, 
            docs, 
            top_k=5
        )

        context = format_docs(docs)

        answer = rag_chain.invoke({
            "context":context,
            "question":q
        })
        print(f'Q: {q}')
        print(f'A: {answer}')
        print('-' * 50)


if __name__ == "__main__":
    main()
