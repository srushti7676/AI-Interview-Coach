
from rag.loader import load_job_description
from rag.retriever import split_documents
from rag.vector_store import create_vector_store, load_vector_store


# Job role and corresponding file paths
JOB_DESCRIPTIONS = {
    "python_developer": "data/job_descriptions/python_developer.txt",
    "backend_developer": "data/job_descriptions/backend_developer.txt",
    "software_developer": "data/job_descriptions/software_developer.txt",
    "machine_learning_engineer": "data/job_descriptions/machine_learning_engineer.txt",
    "data_analyst": "data/job_descriptions/data_analyst.txt",
    "java_developer": "data/job_descriptions/java_developer.txt",
    "frontend_developer": "data/job_descriptions/frontend_developer.txt",
    "full_stack_developer": "data/job_descriptions/full_stack_developer.txt"
}


def index_all_job_descriptions():

    for collection_name, file_path in JOB_DESCRIPTIONS.items():

        print(f"\nIndexing: {collection_name}")

        # Remove the old collection before re-indexing
        existing_store = load_vector_store(collection_name)
        existing_store.delete_collection()

        # Load job description
        documents = load_job_description(file_path)

        # Split into smaller chunks
        chunks = split_documents(documents)

        # Create a separate vector collection
        vector_store = create_vector_store(
            chunks,
            collection_name
        )

        print(
            f"Successfully indexed {len(chunks)} chunks "
            f"for {collection_name}"
        )

    print("\nAll 8 job descriptions indexed successfully!")


if __name__ == "__main__":
    index_all_job_descriptions()
