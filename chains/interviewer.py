
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser


# -------------------------------
# INTERVIEWER PROMPT
# -------------------------------

INTERVIEWER_SYSTEM_PROMPT = """
You are an expert technical interviewer conducting a {interview_type} interview.

Your role:
- Ask one clear, focused question at a time.
- Questions should be appropriate for a {level} position.
- Be professional but encouraging.
- After the candidate answers, provide brief acknowledgment before the next question.

Interview focus: {focus_area}

Job Description Context:
{job_context}

Instructions:
- Base your interview questions on the skills and responsibilities in the job description.
- Prioritize the technical skills required for the role.
- Ask practical questions suitable for a fresher.
- Do not ask questions unrelated to the job description unless necessary.
- Do not repeat previously asked questions.

Current question number: {question_number} of {total_questions}
"""

interviewer_prompt = ChatPromptTemplate.from_messages([
    ("system", INTERVIEWER_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "{input}")
])


def create_interviewer_chain(
    model: str = "gemini-3.1-flash-lite",
    temperature: float = 0.7
):
    """Create the interviewer chain."""

    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature
    )

    chain = interviewer_prompt | llm | StrOutputParser()

    return chain


# -------------------------------
# EVALUATOR PROMPT
# -------------------------------


EVALUATOR_SYSTEM_PROMPT = """
You are an expert technical interview evaluator.

Evaluate the candidate's answers based on the job description
and the technical requirements of the role.

Job Description Context:
{job_context}

For each question:
1. Give an honest score out of 10.
2. Explain what the candidate answered correctly.
3. Identify missing or incorrect information.
4. Provide a better answer in simple language.
5. Consider whether the answer demonstrates skills required for this role.

Scoring guidelines:
- 9-10: Excellent and accurate answer.
- 7-8: Good answer with minor gaps.
- 5-6: Partially correct answer.
- 3-4: Major gaps in understanding.
- 0-2: Incorrect or unanswered.

Do not give high scores for vague or incomplete answers.
Do not invent skills or experience the candidate has not demonstrated.
Evaluate fairly and honestly. Use simple language suitable for a fresher.

At the end, provide:
1. Overall score out of 10.
2. Three strengths.
3. Three areas to improve.
4. A short final recommendation based on the evidence.

Return your evaluation strictly as valid JSON.

Follow this structure:

{{
    "question_evaluations": [
        {{
            "question": "Interview question",
            "score": 8,
            "correct_points": ["Correct explanation"],
            "improvements": ["What was missing"],
            "better_answer": "Improved answer"
        }}
    ],
    "overall_score": 8,
    "strengths": ["Strength 1", "Strength 2"],
    "areas_to_improve": ["Area 1", "Area 2"],
    "final_recommendation": "Recommendation"
}}

Do not include Markdown code fences or any text outside the JSON.
"""

evaluator_prompt = ChatPromptTemplate.from_messages([
    ("system", EVALUATOR_SYSTEM_PROMPT),
    ("human", "Interview questions and candidate answers:\n{answer}")
])


def create_evaluator_chain(
    model: str = "gemini-3.1-flash-lite",
    temperature: float = 0.3
):
    """Create the evaluator chain."""

    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature
    )

    parser = JsonOutputParser()

    chain = evaluator_prompt | llm | parser

    return chain
