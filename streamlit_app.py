import os
import streamlit as st
from dotenv import load_dotenv

# =====================================
# 🌐 Offline Environment Configuration
# =====================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMADB_TELEMETRY"] = "OFF"
os.environ["PYTORCH_JIT_LOG_LEVEL"] = "0"
os.environ["PYTORCH_JIT_LOG_LEVEL_FLAGS"] = "0"
os.environ["STREAMLIT_OFFLINE"] = "true"
os.environ["STREAMLIT_DISABLE_TELEMETRY"] = "true"
os.environ["STREAMLIT_DISABLE_UPDATE_CHECK"] = "true"
os.environ["NO_PROXY"] = "*"
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# =====================================
# 🧠 Load custom modules
# =====================================
from retriever import build_or_load_vectorstore, get_retriever
from qa_chain import build_qa_chain

# Optional local offline loader (no internet)
from langchain.document_loaders import TextLoader

# =====================================
# ⚙️ Page Configuration
# =====================================
st.set_page_config(page_title="Offline Wikipedia Chat", page_icon="🧠", layout="wide")

# =====================================
# 🎨 Custom CSS
# =====================================
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.stApp {
    background-color: #0e1117;
    color: #e1e1e1;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #111418;
    color: #ddd;
    border-right: 1px solid #222;
}
.user-msg {
    background-color: #1e3a8a;
    color: #ffffff;
    padding: 10px 16px;
    border-radius: 16px;
    margin: 8px 0;
    width: fit-content;
    max-width: 80%;
    align-self: flex-end;
    border: 1px solid #2563eb;
    margin-left: auto;
}
.bot-msg {
    background-color: #374151;
    color: #e5e7eb;
    padding: 10px 16px;
    border-radius: 16px;
    margin: 8px 0;
    width: fit-content;
    max-width: 80%;
    border: 1px solid #4b5563;
    margin-right: auto;
}
h1, h2, h3 {
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# 🌍 Environment Setup
# =====================================
load_dotenv()

LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/tinyllama-1.1b-chat.Q4_K_M.gguf")
PERSIST_DIR = os.getenv("PERSIST_DIR", "vectorstore")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
os.makedirs(PERSIST_DIR, exist_ok=True)

# =====================================
# 🧩 Sidebar Configuration
# =====================================
with st.sidebar:
    st.title("⚙️ Wikipedia Indexing (Offline Mode)")
    st.markdown("Use pre-downloaded text files for offline Wikipedia search.")

    use_offline_data = st.checkbox("Use Offline Data", value=True)
    chunk_size = st.number_input("Chunk size", 100, 2000, 500)
    chunk_overlap = st.number_input("Chunk overlap", 0, 500, 50)

    if st.button("📚 Load Local Wikipedia Data"):
        if use_offline_data:
            try:
                # Load from local text file (put your .txt files in 'data/')
                loader = TextLoader("data/wiki_offline.txt", encoding="utf-8")
                docs = loader.load()
                st.success(f"✅ Loaded {len(docs)} offline documents")
                build_or_load_vectorstore(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            except Exception as e:
                st.error(f"⚠️ Failed to load local data: {e}")
        else:
            st.warning("⚠️ Offline mode is enabled. Internet fetch is disabled.")

    st.markdown("---")
    st.markdown("### 🧩 Model Info")
    st.write(f"**Embedding Model:** {EMBED_MODEL}")
    st.write(f"**LLAMA Model Path:** {LLAMA_MODEL_PATH}")
    st.write(f"**Vectorstore Directory:** {PERSIST_DIR}")

# =====================================
# 💬 Chat Section
# =====================================
st.title("💬 Chat with Offline Wikipedia")

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>🧑‍💻 {msg['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>🤖 {msg['text']}</div>", unsafe_allow_html=True)

# Input box
query = st.chat_input("Type your question and press Enter...")

if query:
    retriever = get_retriever(k=3)
    qa = build_qa_chain(retriever, use_memory=False, use_llama=True)

    with st.spinner("🤖 Thinking..."):
        try:
            result = qa(query)
            answer = result.get("result") or result.get("answer") or "Sorry, I couldn’t generate an answer."
        
            # ✅ Format question–answer pairs clearly
            answer = answer.replace("Question:", "\n\n**Question:**").replace("Helpful Answer:", "\n\n**Helpful Answer:**")
        
        except Exception as e:
            answer = f"⚠️ Error: {e}"


    # Append to history
    st.session_state.chat_history.append({"role": "user", "text": query})
    st.session_state.chat_history.append({"role": "assistant", "text": answer})

    st.rerun()

