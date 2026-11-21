from rag.retriever import get_retriever

retriever = get_retriever()

query = "What technical skills are required for the Python Developer role?"

results = retriever.invoke(query)

print("Search Query:", query)
print("\nRetrieved Results:")

for i, document in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(document.page_content)