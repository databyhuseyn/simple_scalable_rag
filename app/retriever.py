from app.vector_db import get_vector_store

def get_retriever():

    vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_type='similarity', 
        search_kwargs={'k':50}
    )