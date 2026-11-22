
import streamlit as st
import re
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage

from chains.interviewer import (
    create_interviewer_chain,
    create_evaluator_chain
)

from rag.retriever import get_retriever

load_dotenv()


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎯",
    layout="wide"
)


# ==========================================
# CUSTOM FRONTEND STYLING
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #F5F7FB;
}

[data-testid="stSidebar"] {
    background-color: #172554;
}

[data-testid="stSidebar"] * {
    color: white;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #172554;
    font-weight: 700;
}

.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
    border: none;
}

.stTextInput input,
.stTextArea textarea {
    border-radius: 8px;
}

/* Sidebar input visibility */

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    color: #172554 !important;
    background-color: white !important;
    -webkit-text-fill-color: #172554 !important;
}

/* Dropdown visibility */

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: white !important;
    color: #172554 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #172554 !important;
}

/* Placeholder visibility */

[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder {
    color: #64748B !important;
    -webkit-text-fill-color: #64748B !important;
}

/* Dropdown options */

[data-baseweb="popover"] [role="option"] {
    color: #172554 !important;
    background-color: white !important;
}

/* Sidebar alerts */

[data-testid="stSidebar"] [data-testid="stAlert"] {
    background-color: #E0F2FE !important;
    border: 1px solid #93C5FD !important;
    border-radius: 8px;
}

[data-testid="stSidebar"] [data-testid="stAlert"] * {
    color: #172554 !important;
    -webkit-text-fill-color: #172554 !important;
}

/* All alerts */

[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    color: #172554 !important;
    -webkit-text-fill-color: #172554 !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HELPER FUNCTION FOR REPORT LISTS
# ==========================================

def display_bullet_list(items):

    if items is None:
        st.caption("No details provided.")
        return

    # Convert different response formats into a list
    if isinstance(items, str):
        raw_items = items.splitlines()

    elif isinstance(items, dict):
        raw_items = list(items.values())

    elif isinstance(items, (list, tuple)):
        raw_items = list(items)

    else:
        raw_items = [items]

    cleaned_items = []

    # Clean each item
    for item in raw_items:

        # Handle nested lists and dictionaries
        if isinstance(item, dict):
            nested_items = list(item.values())

        elif isinstance(item, (list, tuple)):
            nested_items = list(item)

        else:
            nested_items = [item]

        for nested_item in nested_items:

            text = str(nested_item).strip()

            # Remove markdown bullet and numbering symbols
            text = re.sub(
                r"^\s*(?:(?:[-*•▪‣]+)\s*|\d+\s*[:.)]\s*)+",
                "",
                text
            ).strip()

            # Remove empty or unwanted markers
            if text.lower() in {
                "", "-", "*", "•", "svg", "n/a", "na"
            }:
                continue

            cleaned_items.append(text)

    # Display clean bullet points
    if cleaned_items:

        for item in cleaned_items:
            st.markdown(f"- {item}")

    else:

        st.caption("No specific details provided.")


# ==========================================
# INITIALIZE SESSION STATE
# ==========================================

defaults = {
    "interview_started": False,
    "question_number": 0,
    "total_questions": 5,
    "history": [],
    "answers": [],
    "current_question": "",
    "job_context": "",
    "interview_report": None,
    "candidate_name": "",
    "job_role": "Python Developer",
    "level": "Fresher"
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================
# AVAILABLE JOB ROLES
# ==========================================

job_roles = [
    "Python Developer",
    "Backend Developer",
    "Software Developer",
    "Machine Learning Engineer",
    "Data Analyst",
    "Java Developer",
    "Frontend Developer",
    "Full Stack Developer"
]


# ==========================================
# SIDEBAR - INTERVIEW SETUP
# ==========================================

with st.sidebar:

    st.title("🎯 Interview Setup")

    st.write("Prepare for your technical interview.")

    candidate_name = st.text_input(
        "Your Name",
        key="candidate_name_input"
    )

    job_role = st.selectbox(
        "Select Job Role",
        job_roles
    )

    level = st.selectbox(
        "Experience Level",
        ["Fresher", "Intermediate", "Experienced"]
    )

    st.divider()

    start_clicked = False

    if not st.session_state.interview_started:

        start_clicked = st.button(
            "Start Interview",
            use_container_width=True
        )

    elif st.session_state.interview_report is None:

        st.info("Your interview is in progress.")

    else:

        st.success("Interview completed!")


# ==========================================
# START INTERVIEW - GEMINI + RAG
# ==========================================

if start_clicked:

    if not candidate_name.strip():

        st.sidebar.warning("Please enter your name.")

    else:

        try:

            with st.spinner("Preparing your interview..."):

                # Save candidate details
                st.session_state.candidate_name = candidate_name.strip()

                st.session_state.job_role = job_role

                st.session_state.level = level

                # Retrieve selected job description
                retriever = get_retriever(job_role)

                job_documents = retriever.invoke(
                    f"What skills and responsibilities are required "
                    f"for the {job_role} role?"
                )

                job_context = "\n\n".join(
                    document.page_content
                    for document in job_documents
                )

                if not job_context.strip():

                    raise ValueError(
                        "No job description was retrieved from ChromaDB."
                    )

                # Initialize interviewer
                interviewer = create_interviewer_chain()

                # Generate first question
                first_question = interviewer.invoke({

                    "history": [],

                    "interview_type": f"Technical {job_role}",

                    "level": level,

                    "focus_area": job_role,

                    "job_context": job_context,

                    "question_number": 1,

                    "total_questions": st.session_state.total_questions,

                    "input": (
                        "Start the interview by asking the first "
                        "technical question."
                    )
                })

                if not first_question.strip():

                    raise ValueError(
                        "The AI interviewer returned an empty question."
                    )

                # Store interview information
                st.session_state.job_context = job_context

                st.session_state.current_question = first_question

                st.session_state.history = [
                    AIMessage(content=first_question)
                ]

                st.session_state.answers = []

                st.session_state.question_number = 1

                st.session_state.interview_report = None

                st.session_state.interview_started = True

            st.rerun()

        except Exception as e:

            st.sidebar.error(
                f"Unable to start interview: {e}"
            )


# ==========================================
# MAIN PAGE - WELCOME BANNER
# ==========================================

st.html("""
<div style="
    background: linear-gradient(135deg, #172554, #2563EB);
    padding: 35px;
    border-radius: 18px;
    margin-bottom: 25px;
">

    <h1 style="color: white; font-size: 36px;">
        🎯 AI Interview Coach
    </h1>

    <h3 style="color: #DBEAFE;">
        Practice. Improve. Succeed.
    </h3>

    <p style="color: white; font-size: 17px;">
        Prepare for technical interviews with your personal
        AI interviewer. Practice job-specific questions and
        receive instant feedback on your performance.
    </p>

</div>
""")

st.divider()


# ==========================================
# HOME PAGE
# ==========================================

if not st.session_state.interview_started:

    st.header("Welcome!")

    st.write(
        "Your personal AI-powered platform to practise "
        "technical interviews and improve your skills."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.subheader("🤖 AI Interviewer")

            st.write("Practise with Gemini AI.")

    with col2:

        with st.container(border=True):

            st.subheader("📝 Mock Questions")

            st.write("Answer 5 technical questions.")

    with col3:

        with st.container(border=True):

            st.subheader("📊 Performance Report")

            st.write("Get scores and personalized feedback.")

    st.divider()

    st.subheader("How it works")

    st.write("1. Select your job role and experience level.")

    st.write("2. Answer AI-generated technical questions.")

    st.write("3. Receive your interview performance report.")


# ==========================================
# FINAL INTERVIEW REPORT
# ==========================================

elif st.session_state.interview_report is not None:

    st.header("🎉 Interview Completed!")

    report = st.session_state.interview_report

    st.subheader("Your Interview Report")

    # Overall score
    overall_score = report.get("overall_score", 0)

    st.metric(
        "Overall Score",
        f"{overall_score}/10"
    )

    st.divider()

    # Question-wise evaluation
    st.subheader("📋 Question-wise Evaluation")

    question_evaluations = report.get(
        "question_evaluations", []
    )

    expected_questions = len(st.session_state.answers)

    actual_questions = len(question_evaluations)

    if actual_questions != expected_questions:

        st.warning(
            f"Expected {expected_questions} evaluations, "
            f"but received {actual_questions}."
        )

    for index, item in enumerate(
        question_evaluations,
        start=1
    ):

        if not isinstance(item, dict):
            continue

        with st.expander(
            f"Question {index}: {item.get('question', 'Question')}"
        ):

            # Question score
            st.markdown(
                f"**Score: {item.get('score', 'N/A')}/10**"
            )

            # Correct points
            st.markdown("**✅ Correct Points**")

            display_bullet_list(
                item.get("correct_points", [])
            )

            # Question-specific improvements
            st.markdown("**📈 Areas to Improve**")

            display_bullet_list(
                item.get("improvements", [])
            )

            # Better answer
            st.markdown("**💡 Better Answer**")

            better_answer = item.get(
                "better_answer",
                "Not provided."
            )

            if isinstance(better_answer, (list, tuple)):
                display_bullet_list(better_answer)

            else:
                st.write(str(better_answer))

    st.divider()

    # Strengths
    st.subheader("💪 Your Strengths")

    strengths = report.get("strengths", [])

    display_bullet_list(strengths)

    st.divider()

    # Overall improvements
    st.subheader("📈 Overall Areas to Improve")

    improvements = report.get("areas_to_improve", [])

    display_bullet_list(improvements)

    st.divider()

    # Final recommendation
    st.subheader("🏆 Final Recommendation")

    recommendation = report.get(
        "final_recommendation",
        "No recommendation provided."
    )

    st.info(recommendation)

    # Start new interview
    if st.button("🔄 Start New Interview"):

        st.session_state.interview_started = False

        st.session_state.question_number = 0

        st.session_state.history = []

        st.session_state.answers = []

        st.session_state.current_question = ""

        st.session_state.job_context = ""

        st.session_state.interview_report = None

        st.rerun()


# ==========================================
# LIVE INTERVIEW SCREEN
# ==========================================

else:

    st.header("💻 Live Technical Interview")

    # Candidate information
    col1, col2 = st.columns(2)

    with col1:

        st.caption("CANDIDATE")

        st.write(
            f"👤 {st.session_state.candidate_name}"
        )

    with col2:

        st.caption("JOB ROLE")

        st.write(
            f"💼 {st.session_state.job_role}"
        )

    st.divider()

    # Interview progress
    current = st.session_state.question_number

    total = st.session_state.total_questions

    st.markdown(
        f"**Question {current} of {total}**"
    )

    st.progress(current / total)

    # Display question
    with st.container(border=True):

        st.caption("AI INTERVIEWER")

        st.subheader("Your Question")

        st.write(st.session_state.current_question)

    st.divider()

    # Candidate answer form
    with st.form(
        key=f"answer_form_{current}"
    ):

        candidate_answer = st.text_area(
            "Your Answer",
            placeholder="Type your answer here...",
            height=150
        )

        submitted = st.form_submit_button(
            "Submit Answer",
            use_container_width=True
        )

    # Process submitted answer
    if submitted:

        if not candidate_answer.strip():

            st.warning("Please enter your answer.")

        else:

            current_question = st.session_state.current_question

            answer_text = candidate_answer.strip()

            answer_message = HumanMessage(
                content=answer_text
            )

            try:

                if current < total:

                    # Generate next question
                    with st.spinner(
                        "Generating next question..."
                    ):

                        updated_history = (
                            st.session_state.history
                            + [answer_message]
                        )

                        interviewer = create_interviewer_chain()

                        next_question = interviewer.invoke({

                            "history": updated_history,

                            "interview_type": (
                                f"Technical {st.session_state.job_role}"
                            ),

                            "level": st.session_state.level,

                            "focus_area": st.session_state.job_role,

                            "job_context": st.session_state.job_context,

                            "question_number": current + 1,

                            "total_questions": total,

                            "input": (
                                "Ask the next technical interview question. "
                                "Do not repeat previous questions. "
                                "Do not ask closing or general questions."
                            )
                        })

                        if not next_question.strip():

                            raise ValueError(
                                "The AI interviewer returned an empty question."
                            )

                        # Save answer after successful generation
                        st.session_state.answers.append({

                            "question": current_question,

                            "answer": answer_text

                        })

                        st.session_state.history = (
                            updated_history
                            + [AIMessage(content=next_question)]
                        )

                        st.session_state.current_question = next_question

                        st.session_state.question_number += 1

                    st.rerun()

                else:

                    # Evaluate all five answers
                    with st.spinner(
                        "Evaluating your answers..."
                    ):

                        final_answers = (
                            st.session_state.answers
                            + [{
                                "question": current_question,
                                "answer": answer_text
                            }]
                        )

                        if len(final_answers) != total:

                            raise ValueError(
                                "The number of submitted answers "
                                "does not match the interview question count."
                            )

                        # Prepare numbered question-answer pairs
                        all_answers = "\n\n".join(

                            f"QUESTION-ANSWER PAIR {index} "
                            f"OF {len(final_answers)}:\n"
                            f"Question: {item['question']}\n"
                            f"Candidate Answer: {item['answer']}"

                            for index, item in enumerate(
                                final_answers,
                                start=1
                            )
                        )

                        evaluator = create_evaluator_chain()

                        report = evaluator.invoke({

                            "answer": all_answers,

                            "job_context": st.session_state.job_context,

                            "question_count": len(final_answers)

                        })

                        # Validate evaluator response
                        if not isinstance(report, dict):

                            raise ValueError(
                                "The evaluator did not return "
                                "a valid JSON object."
                            )

                        question_evaluations = report.get(
                            "question_evaluations"
                        )

                        if not isinstance(question_evaluations, list):

                            raise ValueError(
                                "The evaluator did not return "
                                "a valid question_evaluations list."
                            )

                        # Save final answers and report
                        st.session_state.answers = final_answers

                        st.session_state.history.append(
                            answer_message
                        )

                        st.session_state.interview_report = report

                    st.rerun()

            except Exception as e:

                st.error(
                    f"Error processing your answer: {e}"
                )
