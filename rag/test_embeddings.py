from rag.loader import load_job_description
from rag.retriever import split_documents
from rag.embeddings import get_embedding_model

file_path = "data/job_descriptions/python_developer.txt"

documents = load_job_description(file_path)

chunks = split_documents(documents)

embedding_model = get_embedding_model()

texts = [chunk.page_content for chunk in chunks]

vectors = embedding_model.embed_documents(texts)

print("Number of vectors:", len(vectors))
print("Vector dimensions:", len(vectors[0]))
print("Embedding test successful!")