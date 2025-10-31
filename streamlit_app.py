import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMADB_TELEMETRY"] = "OFF"
os.environ["PYTORCH_JIT_LOG_LEVEL"] = "0"
os.environ["PYTORCH_JIT_LOG_LEVEL_FLAGS"] = "0"

import streamlit as st
from dotenv import load_dotenv
from wiki_loader import fetch_wikipedia_pages
from retriever import build_or_load_vectorstore, get_retriever
from qa_chain import build_qa_chain
# --- Custom CSS ---
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background-color: #0e1117; /* dark gray background */
        color: #ffffff; /* default text color */
    }

    /* Chat container */
    .chat-message {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }

    /* User message */
    .user-msg {
        background-color: #1e3a8a; /* blue tone */
        color: white;
        border: 1px solid #2563eb;
    }

    /* Bot message */
    .bot-msg {
        background-color: #374151; /* darker gray */
        color: #e5e7eb;
        border: 1px solid #4b5563;
    }

    /* Header or title styling */
    h1, h2, h3 {
        color: #38bdf8; /* cyan highlight */
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ==============================
# Environment setup
# ==============================
load_dotenv()

LLAMA_MODEL_PATH = os.getenv(
    "LLAMA_MODEL_PATH",
    "https://drive.google.com/uc?export=download&id=1Yy3ItSh6tFbGN75_Vd00kX068GN9jESz",
)
PERSIST_DIR = os.getenv("PERSIST_DIR", "vectorstore")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
os.makedirs(PERSIST_DIR, exist_ok=True)

# ==============================
# Streamlit configuration
# ==============================
st.set_page_config(page_title="Offline Wikipedia Chat", page_icon="🧠", layout="wide")

# ==============================
# Custom CSS (ChatGPT-style)
# ==============================
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.stApp {
    background-color: #0e1117;
    color: #e1e1e1;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
}
h1 {
    text-align: center;
    color: #00A67E;
    margin-top: 0.5em;
}
[data-testid="stSidebar"] {
    background-color: #111418;
    color: #ddd;
    border-right: 1px solid #222;
}
.user-msg {
    background-color: #1e1e1e;
    color: #ffffff;
    padding: 10px 16px;
    border-radius: 10px;
    margin: 8px 0;
    width: fit-content;
    max-width: 80%;
    align-self: flex-end;
    border: 1px solid #333;
}
.bot-msg {
    background-color: #202123;
    color: #e1e1e1;
    padding: 10px 16px;
    border-radius: 10px;
    margin: 8px 0;
    width: fit-content;
    max-width: 80%;
    border: 1px solid #333;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Sidebar configuration
# ==============================
with st.sidebar:
    st.title("⚙️ Wikipedia Indexing")
    topic_input = st.text_area(
        "Enter topics (one per line):",
        value="Artificial intelligence\nPython programming\nBlack holes",
        help="Add any Wikipedia topics here!"
    )
    max_pages = st.number_input("Max pages per topic", 1, 10, 2)
    chunk_size = st.number_input("Chunk size", 100, 2000, 500)
    chunk_overlap = st.number_input("Chunk overlap", 0, 500, 50)

    if st.button("📚 Fetch & Index Wikipedia"):
        topics = [t.strip() for t in topic_input.splitlines() if t.strip()]
        with st.spinner("Fetching and indexing Wikipedia pages..."):
            docs = fetch_wikipedia_pages(topics, max_pages_per_topic=max_pages)
        if not docs:
            st.warning("⚠️ No documents fetched. Try different topics or increase max pages.")
        else:
            with st.spinner("🧠 Building local vectorstore..."):
                build_or_load_vectorstore(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.success(f"✅ Indexed {len(docs)} pages successfully!")

    st.markdown("---")
    st.markdown("### 🧩 Model Info")
    st.write(f"**Embedding Model:** {EMBED_MODEL}")
    st.write(f"**LLAMA Model Path:** {LLAMA_MODEL_PATH}")
    st.write(f"**Vectorstore Directory:** {PERSIST_DIR}")

# ==============================
# Main Chat Section
# ==============================
st.title("💬 Chat with Offline Wikipedia")

# Keep chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>🧑‍💻 {msg['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>🤖 {msg['text']}</div>", unsafe_allow_html=True)

# Chat input bar at bottom
query = st.chat_input("Type your question and press Enter...")

if query:
    retriever = get_retriever(k=3)
    qa = build_qa_chain(retriever, use_memory=False, use_llama=True)
    with st.spinner("🤖 Thinking..."):
        try:
            result = qa(query)
            answer = result.get("result") or result.get("answer") or "Sorry, I couldn’t generate an answer."
        except Exception as e:
            answer = f"⚠️ Error: {e}"

    # Update chat
    st.session_state.chat_history.append({"role": "user", "text": query})
    st.session_state.chat_history.append({"role": "assistant", "text": answer})

    # Refresh the chat (new Streamlit syntax)
    st.rerun()
