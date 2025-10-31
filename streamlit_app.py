import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from retriever import build_or_load_vectorstore, get_retriever
from qa_chain import build_qa_chain

# =====================================
# ⚙️ Environment Config
# =====================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMADB_TELEMETRY"] = "OFF"
os.environ["STREAMLIT_DISABLE_TELEMETRY"] = "true"
os.environ["STREAMLIT_DISABLE_UPDATE_CHECK"] = "true"

# =====================================
# 🧠 Load Environment
# =====================================
load_dotenv()
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/tinyllama-1.1b-chat.Q4_K_M.gguf")
PERSIST_DIR = os.getenv("PERSIST_DIR", "vectorstore")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
os.makedirs(PERSIST_DIR, exist_ok=True)

# =====================================
# 🎨 Page Setup
# =====================================
st.set_page_config(page_title="LlamaMind – AI Knowledge Chat", page_icon="🧠", layout="wide")

# =====================================
# 🎨 Custom CSS
# =====================================
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .stApp { background-color: #0e1117; color: #e5e7eb; }
    h1, h2, h3 { color: #38bdf8; }

    .user-msg {
        background-color: #1e3a8a;
        color: #fff;
        padding: 10px 16px;
        border-radius: 16px;
        margin: 8px 0;
        width: fit-content;
        max-width: 80%;
        margin-left: auto;
        border: 1px solid #2563eb;
    }
    .bot-msg {
        background-color: #374151;
        color: #e5e7eb;
        padding: 10px 16px;
        border-radius: 16px;
        margin: 8px 0;
        width: fit-content;
        max-width: 80%;
        margin-right: auto;
        border: 1px solid #4b5563;
    }

    /* Bottom Chat Bar */
    .chat-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #0e1117;
        padding: 15px 20px;
        border-top: 1px solid #333;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 10px;
        z-index: 999;
    }
    .chat-input {
        width: 70%;
        padding: 10px 15px;
        border-radius: 20px;
        border: 1px solid #444;
        background-color: #1e1e1e;
        color: white;
        font-size: 16px;
    }
    .chat-send {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 18px;
        cursor: pointer;
        font-weight: bold;
    }
    .chat-send:hover { background-color: #1d4ed8; }
    .chat-clear {
        background-color: #dc2626;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 14px;
        cursor: pointer;
        font-weight: bold;
    }
    .chat-clear:hover { background-color: #b91c1c; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================
# 🧩 Sidebar
# =====================================
with st.sidebar:
    st.title("⚙️ Wikipedia Indexing")
    use_offline_data = st.checkbox("Use Offline Data", value=True)
    chunk_size = st.number_input("Chunk size", 100, 2000, 500)
    chunk_overlap = st.number_input("Chunk overlap", 0, 500, 50)

    if st.button("📚 Load Local Wikipedia Data"):
        if use_offline_data:
            try:
                loader = TextLoader("data/wiki_offline.txt", encoding="utf-8")
                docs = loader.load()
                st.success(f"✅ Loaded {len(docs)} offline documents")
                build_or_load_vectorstore(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            except Exception as e:
                st.error(f"⚠️ Failed to load local data: {e}")
        else:
            st.warning("⚠️ Offline mode is disabled.")

    st.markdown("---")
    st.markdown("### 🧩 Model Info")
    st.write(f"**Embedding Model:** {EMBED_MODEL}")
    st.write(f"**LLAMA Model Path:** {LLAMA_MODEL_PATH}")
    st.write(f"**Vectorstore Directory:** {PERSIST_DIR}")

# =====================================
# 🧠 Helper – Answer Generation
# =====================================
def generate_answer(question_text):
    retriever = get_retriever(k=1)
    qa = build_qa_chain(retriever, use_memory=False, use_llama=True)
    with st.spinner("🤖 Thinking..."):
        try:
            result = qa(question_text)
            answer = result.get("result") or result.get("answer") or "Sorry, I couldn’t generate an answer."
            answer = answer.replace("Question:", "\n\n**Question:**").replace("Helpful Answer:", "\n\n**Helpful Answer:**")
        except Exception as e:
            answer = f"⚠️ Error: {e}"
    return answer

# =====================================
# 💬 Chat Section
# =====================================
st.title("🤖 LlamaMind – AI Knowledge Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "edit_index" not in st.session_state:
    st.session_state.edit_index = None

# Display messages
for i, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        col1, col2 = st.columns([9, 1])
        with col1:
            st.markdown(f"<div class='user-msg'>🧑‍💻 {msg['text']}</div>", unsafe_allow_html=True)
        with col2:
            if st.button("✏️", key=f"edit_{i}", help="Edit this question"):
                st.session_state.edit_index = i
    else:
        st.markdown(f"<div class='bot-msg'>🤖 {msg['text']}</div>", unsafe_allow_html=True)

# Edit question
if st.session_state.edit_index is not None:
    old_q = st.session_state.chat_history[st.session_state.edit_index]["text"]
    st.info(f"✏️ Editing your question: *{old_q}*")
    new_q = st.text_input("Edit your question:", value=old_q)
    if st.button("🔄 Regenerate Answer"):
        new_ans = generate_answer(new_q)
        st.session_state.chat_history[st.session_state.edit_index]["text"] = new_q
        if (
            st.session_state.edit_index + 1 < len(st.session_state.chat_history)
            and st.session_state.chat_history[st.session_state.edit_index + 1]["role"] == "assistant"
        ):
            st.session_state.chat_history[st.session_state.edit_index + 1]["text"] = new_ans
        else:
            st.session_state.chat_history.append({"role": "assistant", "text": new_ans})
        st.session_state.edit_index = None
        st.rerun()

st.markdown("<div style='height:120px;'></div>", unsafe_allow_html=True)

# =====================================
# 💬 Chat Input (ChatGPT style)
# =====================================
query = st.chat_input("💬 Type your question here...")

# =====================================
# 🚀 Process Input
# =====================================
if query:
    st.session_state.chat_history.append({"role": "user", "text": query})
    answer = generate_answer(query)
    st.session_state.chat_history.append({"role": "assistant", "text": answer})
    st.rerun()
