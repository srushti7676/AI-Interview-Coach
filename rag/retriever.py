
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.vector_store import load_vector_store


def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


def get_retriever(job_role="Python Developer"):

    # Convert job role into its collection name
    collection_name = job_role.strip().lower().replace(" ", "_")

    # Load the corresponding ChromaDB collection
    vector_store = load_vector_store(collection_name)

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    return retriever
