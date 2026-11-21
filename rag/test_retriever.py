from rag.loader import load_job_description
from rag.retriever import split_documents

file_path = "data/job_descriptions/python_developer.txt"

documents = load_job_description(file_path)

chunks = split_documents(documents)

print("Total number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk.page_content)