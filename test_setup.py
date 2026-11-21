
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load the API key from .env
load_dotenv()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

# Send a test message
response = llm.invoke(
    "Say 'Setup complete!' if you can hear me."
)

# Display the response
print(response.content[0]["text"])
