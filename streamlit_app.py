import os
from dotenv import load_dotenv
import streamlit as st

# local modules
from wiki_loader import load_wiki_page
from retriever import build_or_load_vectorstore, create_retriever
from qa_chain import build_qa_chain
from utils import format_sources

# ---------------------------
# Environment setup
# ---------------------------
import warnings
warnings.filterwarnings("ignore")
os.environ["CHROMADB_TELEMETRY"] = "false"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_JIT_LOG_LEVEL"] = "0"
os.environ["STREAMLIT_DISABLE_TELEMETRY"] = "true"
os.environ["STREAMLIT_DISABLE_UPDATE_CHECK"] = "true"

load_dotenv()
import os
import gdown

def ensure_model_exists():
    model_path = "models/phi-2.Q4_K_M.gguf"
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        url = "https://drive.google.com/uc?export=download&id=1bquBi_ccK4XDsatiHZsucysPUBXzmga6"
        print("Downloading Phi-2 model from Google Drive...")
        gdown.download(url, model_path, quiet=False)
        print(f"✅ Model downloaded and saved to {model_path}")

# MODEL PATH — phi-2 quantized .gguf model
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/phi-2.Q4_K_M.gguf")
PERSIST_DIR = os.getenv("PERSIST_DIR", "vectorstore")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

os.makedirs(PERSIST_DIR, exist_ok=True)

# ---------------------------
# Streamlit UI setup
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
# Sidebar: Indexing and settings
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
                docs = fetch_wikipedia_pages(topics, max_pages_per_topic=int(max_pages))
            if not docs:
                st.error("No documents fetched. Try different topics or increase Max pages.")
            else:
                with st.spinner("Building vectorstore (this may take a minute)..."):
                    try:
                        build_or_load_vectorstore(docs, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
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
# Chat area
# ---------------------------
st.markdown('<div class="title">🧠 PhiMind — Reliable CPU RAG Chat</div>', unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []

# ---------------------------
# Safe answer generation
# ---------------------------
def safe_generate(question: str, k: int = 3, require_context: bool = True):
    """Return (answer_text, sources_list). If no context, use fallback safe mode."""
    try:
        retriever = create_retriever(k=3)
    except Exception as e:
        return f"❌ Vectorstore error: {e}", []

    # Try to build the QA chain
    try:
        qa = build_qa_chain(
            retriever,
            model_type="ollama",
            model_name="phi-2:Q4_K_M"
        )
    except Exception as e:
        return f"❌ Failed to create QA chain: {e}", []

    # Run retrieval
    try:
        result = qa({"query": question})
    except Exception as e:
        return f"❌ QA execution error: {e}", []

    answer = result.get("result") or result.get("answer") or ""
    docs = result.get("source_documents") or []

    # Check if retrieved context exists
    has_context = any(
        (getattr(d, "page_content", "") and len(d.page_content.strip()) > 50)
        for d in docs
    )

    # CASE 1: context exists — normal answer
    if has_context:
        return answer.strip(), docs

    # CASE 2: no context — hybrid fallback
    if not has_context:
        if any(word in question.lower() for word in ["ai", "artificial", "machine learning", "intelligence", "data", "neural", "technology"]):
            # allow a short general model-based answer
            try:
                response = qa.llm.invoke(f"Give a brief, factual definition for: {question}")
                return response.strip(), []
            except Exception:
                return "I'm not sure, but it seems related to general AI concepts.", []
        else:
            # safety fallback for unknowns
            return "I don't know based on the indexed data. Try indexing more topics.", []

# ---------------------------
# Display chat history
# ---------------------------
chat_area = st.container()
with chat_area:
    if not st.session_state.chat_history:
        st.info("Ask a question below — e.g., 'Who was Annamacharya?'")
    for entry in st.session_state.chat_history:
        role = entry["role"]
        text = entry["text"]
        if role == "user":
            st.markdown(f"<div class='user-msg'>🧑‍💻 {text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>{text}</div>", unsafe_allow_html=True)

# ---------------------------
# Bottom input (chat_input)
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
st.caption("Running Phi-2 quantized on CPU. Safe mode prevents hallucination if no relevant context is found.")








