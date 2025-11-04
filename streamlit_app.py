import os
import warnings
import subprocess
from dotenv import load_dotenv
import streamlit as st
import gdown  # make sure this is imported
# ---------------------------
# Internal imports (custom project files)
# ---------------------------
from utils.retriever import create_retriever, build_or_load_vectorstore
from utils.qa_chain import build_qa_chain
from utils.helpers import format_sources, ensure_model_exists
from utils.wikipedia_loader import load_wiki_page

# ---------------------------
# Must be FIRST Streamlit command
# ---------------------------
st.set_page_config(page_title="PhiMind – Reliable CPU RAG", page_icon="🧠", layout="wide")

# ---------------------------
# Environment setup
# ---------------------------
warnings.filterwarnings("ignore")
os.environ["CHROMADB_TELEMETRY"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_JIT_LOG_LEVEL"] = "0"
os.environ["STREAMLIT_DISABLE_TELEMETRY"] = "true"
os.environ["STREAMLIT_DISABLE_UPDATE_CHECK"] = "true"

load_dotenv()

# ---------------------------
# Constants and paths
# ---------------------------
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/phi-2.Q4_K_M.gguf")
MODEL_URL = "https://drive.google.com/uc?export=download&id=1bquBi_ccK4XDsatiHZsucysPUBXzmga6"
PERSIST_DIR = os.getenv("PERSIST_DIR", "vectorstore")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

os.makedirs(PERSIST_DIR, exist_ok=True)

# ---------------------------
# Pre-download model before app starts
# ---------------------------
if not os.path.exists(LLAMA_MODEL_PATH):
    st.warning("⚠️ Phi-2 model not found locally. Downloading (~1.7 GB)... Please wait.")
    try:
        gdown.download(MODEL_URL, LLAMA_MODEL_PATH, quiet=False)
        st.success("✅ Phi-2 model downloaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to download Phi-2 model: {e}")
else:
    st.sidebar.success("✅ Phi-2 model already available locally.")


# ---------------------------
# Streamlit page setup
# ---------------------------
st.set_page_config(page_title="PhiMind – Reliable CPU RAG", page_icon="🧠", layout="wide")

st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}
.stApp {background-color:#0b0f14; color:#e6eef6;}
.title {font-size:34px;font-weight:700;color:#38bdf8;margin-bottom:12px;}
.user-msg{background:linear-gradient(180deg,#0b1220,#111827);color:#cfe9ff;
padding:12px 16px;border-radius:14px;border:1px solid rgba(255,255,255,0.03);
margin:8px 0;max-width:75%;margin-left:auto;}
.bot-msg{background:linear-gradient(180deg,#111827,#0b1220);color:#dbeafe;
padding:12px 16px;border-radius:14px;border:1px solid rgba(255,255,255,0.03);
margin:8px 0;max-width:75%;margin-right:auto;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar: Wikipedia Indexing
# ---------------------------
with st.sidebar:
    st.markdown("## 🧭 Wikipedia Indexing")

    topics_input = st.text_area("Topics (one per line)", value="Annamacharya\nRamananda\nQuantum Mechanics")
    max_pages = st.number_input("Max pages per topic", min_value=1, max_value=10, value=2)
    chunk_size = st.number_input("Chunk size", min_value=100, max_value=2000, value=500)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=50)

    if st.button("📚 Fetch & Index Wikipedia"):
        topics = [t.strip() for t in topics_input.splitlines() if t.strip()]
        if not topics:
            st.warning("Please enter at least one topic.")
        else:
            with st.spinner("Fetching Wikipedia pages..."):
                docs = []
                for t in topics:
                    docs.extend(load_wiki_page(t))
            if not docs:
                st.error("No documents fetched. Try different topics or increase Max pages.")
            else:
                with st.spinner("Building vectorstore (this may take a minute)..."):
                    try:
                        build_or_load_vectorstore(docs)
                        st.success(f"Indexed {len(docs)} pages to `{PERSIST_DIR}` ✅")
                    except Exception as e:
                        st.error(f"Failed to build vectorstore: {e}")

    st.markdown("---")
    st.markdown("### ⚙️ Model Info")
    st.write(f"**Model:** Phi-2 (quantized CPU)")
    st.write(f"**Path:** {LLAMA_MODEL_PATH}")
    st.write(f"**Embedding:** {EMBED_MODEL}")
    st.write(f"**Vectorstore:** {PERSIST_DIR}")

    st.markdown("---")
    st.markdown("### Retrieval Settings")
    k_retrieval = st.slider("Retriever k (documents per query)", 1, 6, 3)
    safety_require_context = st.checkbox("Only answer from retrieved context", value=True)

    st.markdown("---")
    if st.button("🧹 Clear chat"):
        st.session_state.chat_history = []
        st.session_state.last_retrieved_docs = []
        st.rerun()


# ---------------------------
# Chat UI
# ---------------------------
st.markdown('<div class="title">🧠 PhiMind — Reliable CPU RAG Chat</div>', unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []


# ---------------------------
# Safe generation function
# ---------------------------
def safe_generate(question: str, k: int = 3, require_context: bool = True):
    """Generate answer safely with automatic model check."""
    if not ensure_model_exists():
        return "❌ Model not found and could not be downloaded.", []

    try:
        retriever = create_retriever(k=k)
    except Exception as e:
        return f"❌ Vectorstore error: {e}", []

    try:
        qa = build_qa_chain(retriever, model_path=LLAMA_MODEL_PATH)
    except Exception as e:
        return f"❌ Failed to create QA chain: {e}", []

    try:
        result = qa({"query": question})
    except Exception as e:
        return f"❌ QA execution error: {e}", []

    answer = result.get("result") or result.get("answer") or ""
    docs = result.get("source_documents") or []

    has_context = any((getattr(d, "page_content", "") and len(d.page_content.strip()) > 50) for d in docs)
    if has_context:
        return answer.strip(), docs
    else:
        return "I don't have enough data to answer that accurately.", []


# ---------------------------
# Display chat messages
# ---------------------------
chat_area = st.container()
with chat_area:
    if not st.session_state.chat_history:
        st.info("Ask a question below — e.g., 'Who was Annamacharya?'")
    for entry in st.session_state.chat_history:
        role, text = entry["role"], entry["text"]
        if role == "user":
            st.markdown(f"<div class='user-msg'>🧑‍💻 {text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>{text}</div>", unsafe_allow_html=True)

# ---------------------------
# Chat input
# ---------------------------
query = st.chat_input("Type your question here...")

if query:
    st.session_state.chat_history.append({"role": "user", "text": query})
    with st.spinner("Thinking..."):
        answer, docs = safe_generate(query, k=k_retrieval, require_context=safety_require_context)
    st.session_state.chat_history.append({"role": "assistant", "text": answer})
    st.session_state.last_retrieved_docs = docs
    st.rerun()

# ---------------------------
# Sources
# ---------------------------
if st.session_state.last_retrieved_docs:
    st.markdown("---")
    st.subheader("📚 Sources")
    try:
        st.markdown(format_sources(st.session_state.last_retrieved_docs))
    except Exception:
        st.warning("Could not format sources.")

st.markdown("---")
st.caption("Running Phi-2 quantized on CPU. If data is missing, the assistant safely declines to answer.")



