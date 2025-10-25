import os
import streamlit as st
from dotenv import load_dotenv
from wiki_loader import fetch_wikipedia_pages
from retriever import build_or_load_vectorstore, get_retriever
from qa_chain import build_qa_chain
from utils import format_sources

# ==============================
# Load environment + setup paths
# ==============================
load_dotenv()

LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "https://drive.google.com/uc?id=1mgbDYKRpG0F4hMpvAZ2uUgnHuTFV21zL"
)
PERSIST_DIR = os.getenv("PERSIST_DIR", "vectorstore")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Create persist dir if not exists
os.makedirs(PERSIST_DIR, exist_ok=True)

# ==============================
# Streamlit UI Configuration
# ==============================
st.set_page_config(page_title="Offline Wikipedia RAG (LLaMA)", layout="wide")
st.title("🧠 Offline Ask Wikipedia — Local LLaMA RAG")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")

    topic_input = st.text_area("Topics to fetch (one per line)", value="Quantum mechanics\nAlbert Einstein")
    max_pages = st.number_input("Max pages per topic", min_value=1, max_value=5, value=2)
    chunk_size = st.number_input("Chunk size", min_value=100, max_value=2000, value=500)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=50)

    if st.button("Fetch & Index Wikipedia"):
        topics = [t.strip() for t in topic_input.splitlines() if t.strip()]
        with st.spinner("📚 Fetching Wikipedia pages..."):
            docs = fetch_wikipedia_pages(topics, max_pages_per_topic=max_pages)

        if not docs:
            st.warning("⚠️ No documents fetched. Try different topics or increase max pages.")
        else:
            with st.spinner("🧠 Building local vectorstore..."):
                build_or_load_vectorstore(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.success(f"✅ Indexed {len(docs)} pages successfully!")

st.markdown("---")

# ==============================
# Session State for Chat Memory
# ==============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of (question, answer)

# ==============================
# Main Input Section
# ==============================
query = st.text_input("💬 Ask a question about your indexed topics:")
k = st.slider("Number of retrieved chunks (k)", min_value=1, max_value=10, value=3)

if st.button("Ask Question") and query.strip():
    try:
        retriever = get_retriever(k=k)
    except Exception as e:
        st.error(f"❌ Could not load vectorstore: {e}. Please index topics first.")
        retriever = None

    if retriever:
        try:
            qa = build_qa_chain(
                retriever,
                use_memory=False,
                use_llama=True,
                model_path=LLAMA_MODEL_PATH
            )

            with st.spinner("🤖 Generating answer... please wait..."):
                result = qa(query)

            answer = result.get("result") or result.get("answer")
            docs = result.get("source_documents", [])

            # Save chat
            st.session_state.chat_history.append((query, answer))

            # Display answer
            st.subheader("💡 Answer")
            st.write(answer)

            # Display sources
            st.subheader("📚 Sources")
            st.markdown(format_sources(docs))

        except Exception as e:
            st.error(f"Error during QA generation: {e}")

st.markdown("---")

# ==============================
# Show Chat History
# ==============================
if st.session_state.chat_history:
    st.subheader("🗂️ Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
        st.markdown("---")

# ==============================
# Model Info Display
# ==============================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧩 Model Info")
st.sidebar.write(f"**Embedding Model:** {EMBED_MODEL}")
st.sidebar.write(f"**LLAMA Model Path:** {LLAMA_MODEL_PATH}")
st.sidebar.write(f"**Vectorstore Directory:** {PERSIST_DIR}")
