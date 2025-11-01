import os
from dotenv import load_dotenv
import streamlit as st

# Local modules (must exist in your repo)
from wiki_loader import fetch_wikipedia_pages
from retriever import build_or_load_vectorstore, get_retriever
from qa_chain import build_qa_chain
from utils import format_sources

# -------------------------
# Environment / safety flags
# -------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMADB_TELEMETRY"] = "OFF"
os.environ["PYTORCH_JIT_LOG_LEVEL"] = "0"
os.environ["PYTORCH_JIT_LOG_LEVEL_FLAGS"] = "0"
os.environ["STREAMLIT_DISABLE_TELEMETRY"] = "true"
os.environ["STREAMLIT_DISABLE_UPDATE_CHECK"] = "true"

load_dotenv()

LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/phi-2.Q4_K_M.gguf")
PERSIST_DIR = os.getenv("PERSIST_DIR", "vectorstore")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

os.makedirs(PERSIST_DIR, exist_ok=True)

# -------------------------
# Page config + CSS
# -------------------------
st.set_page_config(page_title="LlamaMind – AI Knowledge Chat", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    /* hide streamlit menu/footer */
    #MainMenu, footer {visibility: hidden;}

    /* page background */
    .stApp { background-color: #0b0f14; color: #e6eef6; }

    /* header */
    .title { font-size: 34px; font-weight: 700; color: #38bdf8; margin-bottom: 12px; }

    /* chat bubbles */
    .user-msg {
        background: linear-gradient(180deg,#0b1220,#111827);
        color: #cfe9ff;
        padding: 12px 16px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.03);
        margin: 8px 0;
        max-width:75%;
        margin-left: auto;
    }
    .bot-msg {
        background: linear-gradient(180deg,#111827,#0b1220);
        color: #dbeafe;
        padding: 12px 16px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.03);
        margin: 8px 0;
        max-width:75%;
        margin-right: auto;
    }

    /* fixed bottom input bar */
    .bottom-bar {
        position: fixed;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 999;
        background: #071018;
        padding: 12px 16px;
        border-top: 1px solid rgba(255,255,255,0.03);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .bottom-input {
        width: 78%;
        background: #0f1724;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.03);
        padding: 10px 14px;
        color: #e6eef6;
    }
    .send-btn {
        background: #0ea5a3;
        color: #062023;
        border: none;
        padding: 10px 14px;
        border-radius: 10px;
        font-weight: 700;
        cursor: pointer;
    }
    .send-btn:active { transform: translateY(1px); }
    .small-btn { background: #111827; color: #dbeafe; padding: 8px 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.03); cursor:pointer; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Sidebar: indexing & model info
# -------------------------
with st.sidebar:
    st.markdown("## 🧭 Wikipedia Indexing")
    st.markdown("Enter topics (one per line), then Fetch & Index to build the vectorstore.")
    topics_input = st.text_area("Topics (one per line)", value="Albert Einstein\nQuantum mechanics")
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
                        st.success(f"Indexed {len(docs)} pages to `{PERSIST_DIR}`")
                    except Exception as e:
                        st.error(f"Failed to build vectorstore: {e}")

    st.markdown("---")
    st.markdown("### 🧩 Model & Vectorstore")
    st.write(f"**Embedding model:** {EMBED_MODEL}")
    st.write(f"**LLAMA model path:** {LLAMA_MODEL_PATH}")
    st.write(f"**Vectorstore dir:** {PERSIST_DIR}")

    st.markdown("---")
    st.markdown("### ⚙️ Options")
    k_retrieval = st.slider("Retriever k (how many docs to retrieve)", 1, 6, 3)
    safety_require_context = st.checkbox("Only answer from retrieved context (preferred)", value=True)

# -------------------------
# Main area (title + chat view)
# -------------------------
st.markdown('<div class="title">🤖 LlamaMind — AI Knowledge Chat</div>', unsafe_allow_html=True)

# initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # each item: {"role":"user"/"assistant", "text":...}
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# helper: safe generate using retriever + Llama chain
def safe_generate(question: str, k: int = 3, require_context: bool = True):
    """Return (answer_text, sources_list). If require_context and no context -> safe 'I don't know'."""
    try:
        retriever = get_retriever(k=k)
    except Exception as e:
        return f"❌ Vectorstore error: {e}", []

    # Build QA chain using qa_chain helper (this uses local LLaMA if configured)
    try:
        qa = build_qa_chain(retriever, use_memory=False, use_llama=True)
    except FileNotFoundError as fe:
        # model missing — surface friendly message
        return (f"❌ LLaMA model not found at '{LLAMA_MODEL_PATH}'.\n"
                "Put the model file path in .env (LLAMA_MODEL_PATH) or download the model file."), []
    except Exception as e:
        return f"❌ Failed to create QA chain: {e}", []

    # first do a quick retrieval-only check to see whether there is context
    try:
        # Some RetrievalQA wrappers return source_documents in the result; call chain
        result = qa({"query": question})
    except TypeError:
        # older/newer langchain versions sometimes accept (question) directly
        try:
            result = qa(question)
        except Exception as e:
            return f"❌ QA execution error: {e}", []
    except Exception as e:
        return f"❌ QA execution error: {e}", []

    # Extract answer and sources
    answer = result.get("result") or result.get("answer") or ""
    docs = result.get("source_documents") or result.get("source_documents", []) or []

    # If require_context and docs is empty -> don't let model hallucinate
    if require_context:
        # check if any retrieved doc looks meaningful (basic check)
        has_content = False
        if isinstance(docs, (list, tuple)) and len(docs) > 0:
            # require at least one doc with >50 chars
            for d in docs:
                cnt = getattr(d, "page_content", None) or getattr(d, "content", None) or ""
                if isinstance(cnt, str) and len(cnt.strip()) > 50:
                    has_content = True
                    break

        if not has_content:
            return "I don't know based on the indexed Wikipedia pages. Try indexing more topics or increase 'k'.", []

    # return final text and docs
    return answer.strip(), docs

# -------------------------
# show chat history
# -------------------------
chat_area = st.container()
with chat_area:
    if not st.session_state.chat_history:
        st.info("Ask a question using the input at the bottom. Try: 'Who is Alan Turing?'")
    for entry in st.session_state.chat_history:
        role = entry.get("role")
        text = entry.get("text")
        if role == "user":
            st.markdown(f"<div class='user-msg'>🧑‍💻 {st.markdown(text, unsafe_allow_html=False) or text}</div>", unsafe_allow_html=True)
        else:
            # assistant
            st.markdown(f"<div class='bot-msg'>{text}</div>", unsafe_allow_html=True)

# store last retrieved docs to optionally show below input
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []

# -------------------------
# bottom input bar (chat_input works with Enter)
# -------------------------
# Use columns so the input sits centered and a send button exists as well
st.markdown("<div style='height:90px'></div>", unsafe_allow_html=True)  # spacer so bottom bar doesn't overlay content

# streamlit's chat_input returns when user presses Enter or the send icon
query = st.chat_input("Type your question here — press Enter or click Send...")

# Also provide a clear chat small button in the sidebar or here
if st.sidebar.button("🧹 Clear chat"):
    st.session_state.chat_history = []
    st.session_state.last_retrieved_docs = []

if query:
    st.session_state.last_query = query
    # append user message
    st.session_state.chat_history.append({"role": "user", "text": query})

    # generate safely
    safe_ans, docs = safe_generate(query, k=int(k_retrieval), require_context=bool(safety_require_context))

    # show short spinner and append assistant
    st.session_state.chat_history.append({"role": "assistant", "text": safe_ans})
    st.session_state.last_retrieved_docs = docs

    # Scroll will update on next rerun/render automatically

# Optional: show sources below chat if available
if st.session_state.last_retrieved_docs:
    st.markdown("---")
    st.subheader("📚 Sources (retrieved chunks)")
    try:
        st.markdown(format_sources(st.session_state.last_retrieved_docs))
    except Exception:
        st.write("Could not format sources (different doc shape).")

# -------------------------
# Helpful note / troubleshooting
# -------------------------
st.markdown("---")
st.caption(
    "Notes: If the local LLaMA model file is missing the app will show an explanatory message. "
    "If responses seem short or cut off, lower `max_new_tokens` in your `qa_chain.py` or "
    "increase `n_ctx` depending on the model. To reduce hallucinations set a lower `temperature` in your model config."
)
