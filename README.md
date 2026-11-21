# AI-Powered Job-Specific Interview Coach Using RAG

## Project Overview

An AI-powered interview coaching application that conducts technical mock interviews based on job descriptions. It uses Google Gemini and Retrieval-Augmented Generation (RAG) to generate relevant interview questions and evaluate candidate responses.

## Features

- AI-generated technical interview questions.
- Job-description-based question generation using RAG.
- Conversation history stored using JSON.
- Automated answer evaluation and scoring.
- Personalized feedback, strengths, and areas for improvement.
- Final interview evaluation report.

## Technologies Used

- Python
- LangChain
- Google Gemini
- Gemini Embeddings
- ChromaDB
- Retrieval-Augmented Generation (RAG)
- JSON

## Project Workflow

1. Load the job description from a text file.
2. Split the document into smaller chunks.
3. Generate embeddings using Gemini.
4. Store embeddings in ChromaDB.
5. Retrieve relevant job-description information using similarity search.
6. Generate technical interview questions using Gemini.
7. Store interview questions and candidate answers in conversation memory.
8. Evaluate answers and generate a final report.

## Project Structure

```text
interview_coach/
│
├── main.py
├── chains/
├── memory/
├── rag/
├── agents/
├── data/
│   └── job_descriptions/
├── .env
├── .gitignore
└── README.md
```

## How to Run

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the required Python dependencies.
4. Configure your Google Gemini API key in a local `.env` file.
5. Run the application:

```bash
python main.py
```

## Future Improvements

- Develop a web-based user interface.
- Support multiple job descriptions and PDF/DOCX uploads.
- Implement adaptive interview difficulty.
- Add user authentication and separate interview sessions.
- Develop REST APIs for frontend integration.

## Author

Srushti S