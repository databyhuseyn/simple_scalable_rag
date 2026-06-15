from qdrant_client import QdrantClient
from app.embedder import embeddings
from app.chunker import chunk_documents
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
import glob

client = QdrantClient(
    host='localhost',
    port='6333'
)


def store_vectors(chunks):
    
    vector_store = QdrantVectorStore.from_documents(
        collection_name='pdf_docs',
        embedding=embeddings,
        sparse_embedding=FastEmbedSparse(
            model_name="Qdrant/bm25"
        ),
        retrieval_mode = RetrievalMode.HYBRID,
        url='localhost:6333',
        documents=chunks
    )

    return vector_store

def get_vector_store():

    return QdrantVectorStore(
        client=client,
        collection_name='pdf_docs',
        embedding=embeddings,
        sparse_embedding=FastEmbedSparse(
            model_name="Qdrant/bm25"
        ),
        retrieval_mode = RetrievalMode.HYBRID
    )