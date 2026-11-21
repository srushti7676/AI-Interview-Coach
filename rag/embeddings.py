from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def get_embedding_model():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    return embeddings