from langchain_chroma import Chroma
from rag.embeddings import get_embedding_model


def create_vector_store(chunks, collection_name="job_descriptions"):

    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory="chroma_db"
    )

    return vector_store


def load_vector_store(collection_name="job_descriptions"):

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory="chroma_db"
    )

    return vector_store
