from app.document_loaders import pdf_loader
from app.chunker import chunk_documents
from app.vector_db import store_vectors

def ingest_documents():
    pdfs = pdf_loader()
    chunks = chunk_documents(pdfs)
    store_vectors(chunks)

    print("Ingestion Completed!")


ingest_documents()