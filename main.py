
from dotenv import load_dotenv
from chains.interviewer import create_interviewer_chain, create_evaluator_chain
from langchain_core.messages import HumanMessage, AIMessage
from memory.conversation import ConversationMemory
from rag.retriever import get_retriever

load_dotenv()

interviewer = create_interviewer_chain()
evaluator = create_evaluator_chain()

# Create conversation memory
memory = ConversationMemory()

# Load previous conversations
history = []

for message in memory.get_history():
    if message["role"] == "interviewer":
        history.append(AIMessage(content=message["content"]))
    elif message["role"] == "candidate":
        history.append(HumanMessage(content=message["content"]))

# Retrieve relevant job description context
retriever = get_retriever()

job_documents = retriever.invoke(
    "What skills, technologies, and responsibilities are required for this Python Developer role?"
)

job_context = "\n\n".join(
    document.page_content for document in job_documents
)

print("\nJob description loaded successfully!")

questions = []
answers = []

for question_number in range(1, 6):

    response = interviewer.invoke({
        "history": history,
        "interview_type": "technical Python",
        "level": "fresher",
        "focus_area": "Python fundamentals",
        "job_context": job_context,
        "question_number": question_number,
        "total_questions": 5,
        "input": "Ask me the next interview question. Do not repeat previous questions."
    })

    print(f"\nInterviewer: {response}")

    # Save the interviewer's question
    history.append(AIMessage(content=response))
    memory.add_message("interviewer", response)
    questions.append(response)

    answer = input("You: ")

    # Save the candidate's answer
    history.append(HumanMessage(content=answer))
    memory.add_message("candidate", answer)
    answers.append(answer)


print("\nInterview completed! Evaluating your answers...")

all_answers = ""

for i in range(5):
    all_answers += f"""
Question {i+1}: {questions[i]}
Candidate Answer: {answers[i]}
"""

feedback = evaluator.invoke({
    "answer": all_answers,
    "job_context": job_context
})

print("\n========== FINAL INTERVIEW REPORT ==========")

print("\nOverall Score:", feedback["overall_score"], "/ 10")

print("\n========== QUESTION-WISE EVALUATION ==========")

for item in feedback["question_evaluations"]:

    print("\nQuestion:", item["question"])
    print("Score:", item["score"], "/ 10")

    print("Correct Points:")
    for point in item["correct_points"]:
        print("-", point)

    print("Areas to Improve:")
    for improvement in item["improvements"]:
        print("-", improvement)

    print("Better Answer:")
    print(item["better_answer"])


print("\n========== STRENGTHS ==========")

for strength in feedback["strengths"]:
    print("-", strength)


print("\n========== AREAS TO IMPROVE ==========")

for area in feedback["areas_to_improve"]:
    print("-", area)


print("\n========== FINAL RECOMMENDATION ==========")
print(feedback["final_recommendation"])
