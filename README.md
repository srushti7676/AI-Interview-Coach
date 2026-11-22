# AI-Powered Job-Specific Interview Coach Using RAG

## Project Overview

An AI-powered interview coaching application that conducts personalized technical mock interviews based on selected job roles and their job descriptions.

The application uses Google Gemini, LangChain, Retrieval-Augmented Generation (RAG), and ChromaDB to generate relevant technical interview questions, evaluate candidate responses, and provide personalized feedback.

It includes an interactive web interface built using Streamlit, allowing candidates to practise interviews and receive performance reports.

## Features

- Interactive web interface built with Streamlit.
- Supports 8 job roles:
  - Python Developer
  - Backend Developer
  - Software Developer
  - Machine Learning Engineer
  - Data Analyst
  - Java Developer
  - Frontend Developer
  - Full Stack Developer
- Role-specific technical interview question generation.
- Job-description-based question retrieval using RAG.
- Separate ChromaDB collections for each job role.
- Five-question technical mock interview sessions.
- Candidate answer submission through the web interface.
- Automated answer evaluation using Google Gemini.
- Individual question scores and detailed feedback.
- Personalized strengths and areas for improvement.
- Overall interview score and final recommendation.
- Conversation history support using JSON-based memory in the terminal application.
- Option to start a new interview.

## Technologies Used

- Python
- Streamlit
- LangChain
- Google Gemini
- Gemini Embeddings
- ChromaDB
- Retrieval-Augmented Generation (RAG)
- JSON
- Git and GitHub

## Project Workflow

1. Load job descriptions from text files.
2. Split documents into smaller chunks.
3. Generate embeddings using Gemini.
4. Store embeddings in separate ChromaDB collections for each job role.
5. Retrieve relevant job-description information using similarity search.
6. Allow candidates to select their preferred job role and experience level.
7. Generate five role-specific technical interview questions using Gemini.
8. Collect candidate answers through the Streamlit interface.
9. Evaluate answers using the AI evaluator.
10. Generate a final interview report containing individual scores, strengths, areas for improvement, and a recommendation.

## Project Structure

```text
interview_coach/
│
├── app.py
├── main.py
│
├── chains/
│   └── interviewer.py
│
├── memory/
│
├── rag/
│   ├── embeddings.py
│   ├── loader.py
│   ├── retriever.py
│   ├── vector_store.py
│   └── index_job_descriptions.py
│
├── agents/
│
├── data/
│   └── job_descriptions/
│       ├── python_developer.txt
│       ├── backend_developer.txt
│       ├── software_developer.txt
│       ├── machine_learning_engineer.txt
│       ├── data_analyst.txt
│       ├── java_developer.txt
│       ├── frontend_developer.txt
│       └── full_stack_developer.txt
│
├── .env
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/srushti7676/AI-Interview-Coach.git
```

### 2. Navigate to the Project Directory

```bash
cd AI-Interview-Coach
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install streamlit python-dotenv langchain-core langchain-google-genai langchain-community langchain-text-splitters langchain-chroma
```

### 6. Configure Google Gemini API Key

Create a `.env` file in the project root directory and add your Google Gemini API key:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

Keep your API key private. Do not upload your `.env` file to GitHub.

### 7. Index Job Descriptions

Run the following command to generate embeddings and create separate ChromaDB collections for all eight job roles:

```bash
python -m rag.index_job_descriptions
```

### 8. Run the Streamlit Application

```bash
streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

### 9. Run the Terminal-Based Application

The original terminal-based interview application can also be started using:

```bash
python main.py
```

## Future Enhancements

- Support PDF and DOCX job-description uploads.
- Implement adaptive interview difficulty.
- Add user authentication and personalized interview history.
- Develop detailed performance analytics and dashboards.
- Add downloadable interview evaluation reports.
- Improve evaluation consistency and question-wise scoring.

## Author

Srushti S