from rag.loader import load_job_description
from rag.retriever import split_documents
from rag.vector_store import create_vector_store

file_path = "data/job_descriptions/python_developer.txt"

documents = load_job_description(file_path)

chunks = split_documents(documents)

vector_store = create_vector_store(chunks)

print("Vector database created successfully!")
print("Total documents stored:", vector_store._collection.count())