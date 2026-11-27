import os
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from groq import Groq

from rag import StudentRAG

BASE = os.path.dirname(os.path.abspath(__file__))

# Page setup
st.set_page_config(
    page_title="Student Assistant with RAG",
    page_icon="🎓",
    layout="wide"
)

# Header
st.title("🎓 Student Assistant ")
st.caption("Sapthagiri NPS University | Section 3ADS01 | Semester III | Powered by Groq & RAG")


def get_api_key():
    # 1. Streamlit secrets
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    # 2. Environment variable
    return os.environ.get("GROQ_API_KEY")


# Sidebar Settings
st.sidebar.header("⚙️ Settings & Configuration")

api_key = get_api_key()
if not api_key:
    api_key = st.sidebar.text_input("Enter your Groq API key", type="password")

if not api_key:
    st.info("👈 Please enter your Groq API key in the sidebar to get started.")
    st.stop()

# LLM Model Selector
AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b"
]
selected_model = st.sidebar.selectbox("LLM Model", AVAILABLE_MODELS, index=0)

client = Groq(api_key=api_key)


# Initialize RAG Engine
@st.cache_resource(show_spinner="Loading RAG Index & Embedding Model...")
def load_rag_pipeline():
    rag = StudentRAG()

    # Preload default files
    timetable_path = os.path.join(BASE, "timetable.txt")
    if os.path.exists(timetable_path):
        with open(timetable_path, "r", encoding="utf-8") as f:
            rag.add_raw_text(f.read(), source_name="Timetable (3ADS01)")

    college_info_path = os.path.join(BASE, "college_info.txt")
    if os.path.exists(college_info_path):
        with open(college_info_path, "r", encoding="utf-8") as f:
            rag.add_raw_text(f.read(), source_name="College Guidelines")

    return rag


rag_engine = load_rag_pipeline()


# Sidebar: Document Uploader
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Upload College Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload Circulars, Syllabus, or Notes (.pdf, .txt, .md)",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

if uploaded_files:
    if "uploaded_doc_names" not in st.session_state:
        st.session_state.uploaded_doc_names = set()

    for file in uploaded_files:
        if file.name not in st.session_state.uploaded_doc_names:
            added_chunks = rag_engine.add_document(file, filename=file.name)
            st.session_state.uploaded_doc_names.add(file.name)
            st.sidebar.success(f"✅ Indexed `{file.name}` ({added_chunks} chunks)")


# Sidebar: Knowledge Base Status
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Knowledge Base Status")
sources = sorted(list(set(chunk["source"] for chunk in rag_engine.chunks)))
st.sidebar.metric(label="Total Indexed Chunks", value=len(rag_engine.chunks))
if sources:
    st.sidebar.write("**Active Sources:**")
    for s in sources:
        st.sidebar.markdown(f"- 📄 `{s}`")

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()


# Quick Sample Questions for Demonstration
st.markdown("##### 💡 Try asking:")
cols = st.columns(4)
sample_prompts = [
    "What is my next class today?",
    "What is the minimum attendance required?",
    "Who is the faculty for DBMS?",
    "How to get a new ID card?"
]
for i, col in enumerate(cols):
    if col.button(sample_prompts[i], key=f"sample_{i}"):
        st.session_state.active_prompt = sample_prompts[i]


def build_system_prompt(retrieved_context: str):
    now = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%A, %d %B %Y, %I:%M %p")
    return f"""You are Jarvis, an AI student assistant for Sapthagiri NPS University, section 3ADS01 (Semester III, Artificial Intelligence and Data Science).
Use a formal, polite, supportive and professional tone at all times. Do not use slang or casual language.

CRITICAL INSTRUCTIONS FOR ANSWERING:
1. FULL & COMPLETE ANSWERS:
   - When answering questions from uploaded materials (e.g. Question Banks, Notes, Syllabus, Assignments), provide the FULL, THOROUGH, and COMPLETE answer.
   - If the student's question has multiple sub-parts (e.g., part a, b, c or multiple sub-questions), answer EVERY single part thoroughly using the retrieved context. Do NOT cut short or prematurely summarize if the details are available in the context.

2. STRICT HANDLING WHEN INFORMATION IS MISSING (Distinguish carefully):
   A) For ACADEMIC & UPLOADED DOCUMENT QUERIES (Subject concepts, question bank solutions, technical questions):
      - If you cannot find the answer (or cannot find a specific part of a multi-part question) in the retrieved context, state clearly:
        "I could not find the answer to this question [or this specific part] in the uploaded documents."
      - NEVER direct academic, technical, or question bank queries to the class advisor Mr. Manjunath M unless the uploaded document explicitly designates him for that subject.
   
   B) For COLLEGE ADMINISTRATIVE & GUIDELINE QUERIES (Timetable changes, attendance shortages/medical leave, fee queries, ID cards, internal assessment IA schedule):
      - If the exact administrative details are unannounced or missing, answer what is known and state:
        "Please consult your class advisor, Mr. Manjunath M, or the college office for further clarification."

3. TIMETABLE & REAL-TIME SCHEDULING:
   - Use CURRENT DATE AND TIME to determine what "today", "tomorrow", and "next class" mean. Saturday and Sunday have no scheduled classes.
   - Use full course names, codes, and faculty names when helpful.

4. GROUNDING:
   - Rely strictly on the information provided in the RETRIEVED CONTEXT below. Never fabricate facts.

CURRENT DATE AND TIME: {now}

=== RETRIEVED CONTEXT (RAG) ===
{retrieved_context if retrieved_context else "No relevant context found in documents."}
"""


if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# Capture user input from either chat input or sample prompt buttons
prompt = st.chat_input("Ask about your timetable, syllabus, or college...")
if "active_prompt" in st.session_state and st.session_state.active_prompt:
    prompt = st.session_state.active_prompt
    st.session_state.active_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1. Retrieval Step (RAG)
    with st.spinner("Retrieving relevant knowledge..."):
        retrieved_chunks = rag_engine.retrieve(prompt, top_k=6)

    # Assemble retrieved context
    context_blocks = []
    for c in retrieved_chunks:
        context_blocks.append(f"[Source: {c['source']}]\n{c['content']}")
    retrieved_context_str = "\n\n".join(context_blocks)

    # 2. Generation Step (LLM)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-10:]
        if m["role"] in ["user", "assistant"]
    ]
    api_messages = [{"role": "system", "content": build_system_prompt(retrieved_context_str)}] + history

    with st.chat_message("assistant"):
        try:
            with st.spinner("Generating answer..."):
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=api_messages,
                    temperature=0.2,
                    max_tokens=2048
                )
            reply = response.choices[0].message.content
            st.markdown(reply)
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply
            })
        except Exception as e:
            st.error(f"Sorry, something went wrong: {e}")
            st.session_state.messages.pop()
