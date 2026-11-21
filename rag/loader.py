
from langchain_community.document_loaders import TextLoader


def load_job_description(file_path):
    """
    Load a job description from a text file.
    """

    loader = TextLoader(file_path, encoding="utf-8")

    documents = loader.load()

    return documents
