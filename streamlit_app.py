import os
import streamlit as st
from dotenv import load_dotenv
from wiki_loader import fetch_wikipedia_pages
from retriever import build_or_load_vectorstore, get_retriever
from qa_chain import build_qa_chain
from utils import format_sources

# Load environment
load_dotenv()

# Streamlit config
st.set_page_config(page_title="Offline Wikipedia RAG (LLaMA)", layout="wide")
st.title("🧠 Offline Ask Wikipedia — Local LLaMA RAG")

# Sidebar setup
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
            st.warning("No documents fetched. Try different topics.")
        else:
            with st.spinner("🧠 Building local vectorstore..."):
                build_or_load_vectorstore(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.success(f"✅ Indexed {len(docs)} pages successfully!")

st.markdown("---")

# Initialize session state for memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of (question, answer)

# Main section
query = st.text_input("Ask a question about your indexed topics:")
k = st.slider("Number of retrieved chunks (k)", min_value=1, max_value=10, value=3)

if st.button("Ask Question") and query.strip():
    try:
        retriever = get_retriever(k=k)
    except Exception as e:
        st.error(f"Could not load vectorstore: {e}. Please index topics first.")
        retriever = None

    if retriever:
        qa = build_qa_chain(retriever, use_memory=False, use_llama=True)
        with st.spinner("🤖 Generating answer..."):
            result = qa(query)

        answer = result.get("result") or result.get("answer")
        docs = result.get("source_documents", [])

        # Save to chat history
        st.session_state.chat_history.append((query, answer))

        # Display current answer
        st.subheader("💡 Answer")
        st.write(answer)

        # Display sources
        st.subheader("📚 Sources")
        st.markdown(format_sources(docs))

st.markdown("---")

# Show full conversation history
if st.session_state.chat_history:
    st.subheader("🗂️ Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
        st.markdown("---")
