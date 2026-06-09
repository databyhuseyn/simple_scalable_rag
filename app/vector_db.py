from qdrant_client import QdrantClient
from app.embedder import embeddings
from app.chunker import chunk_documents
from langchain_qdrant import QdrantVectorStore
import glob

client = QdrantClient(
    host='localhost',
    port='6333'
)


def store_vectors(chunks):
    

    vector_store = QdrantVectorStore.from_documents(
        collection_name='pdf_docs',
        embedding=embeddings,
        url='localhost:6333',
        documents=chunks
    )

    return vector_store

