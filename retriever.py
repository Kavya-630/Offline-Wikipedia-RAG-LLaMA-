import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader


# =====================================
# ⚙️ Default Config
# =====================================
PERSIST_DIR = os.getenv("PERSIST_DIR", "vectorstore")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_PATH = os.getenv("DATA_PATH", "data/wiki_offline.txt")


# =====================================
# 📚 Load and Chunk Documents
# =====================================
def load_documents(data_path: str = DATA_PATH):
    """
    Load text documents from a local .txt file.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"❌ Data file not found at {data_path}. Please ensure 'wiki_offline.txt' exists."
        )

    print(f"📄 Loading data from: {data_path}")
    loader = TextLoader(data_path, encoding="utf-8")
    docs = loader.load()
    return docs


# =====================================
# ✂️ Split Text into Chunks
# =====================================
def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    """
    Split text into overlapping chunks for embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    print(f"✂️ Splitting documents into chunks of {chunk_size} with overlap {chunk_overlap}")
    split_docs = text_splitter.split_documents(docs)
    print(f"✅ Created {len(split_docs)} chunks.")
    return split_docs


# =====================================
# 🧩 Build or Load Vectorstore
# =====================================
def build_retriever(chunk_size=500, chunk_overlap=50, k=3):
    """
    Build or load a Chroma vectorstore retriever.
    """
    os.makedirs(PERSIST_DIR, exist_ok=True)

    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Check if Chroma DB already exists
    if os.path.exists(os.path.join(PERSIST_DIR, "index")) or len(os.listdir(PERSIST_DIR)) > 0:
        print(f"🔄 Loading existing Chroma vectorstore from '{PERSIST_DIR}'...")
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
    else:
        print("🚀 Building new Chroma vectorstore...")
        docs = load_documents()
        chunks = chunk_documents(docs, chunk_size, chunk_overlap)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=PERSIST_DIR,
        )
        vectorstore.persist()
        print(f"✅ Vectorstore created and saved to '{PERSIST_DIR}'")

    # use `k` for retriever search kwargs
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    print(f"🔍 Retriever ready with top-{k} search.")
    return retriever

# =====================================
# 🧱 Backward Compatibility
# =====================================
build_or_load_vectorstore = build_retriever
get_retriever = build_retriever


# =====================================
# 🧪 Debug Run
# =====================================
if __name__ == "__main__":
    print("🔧 Testing retriever building...")
    retriever = build_retriever()
    print("✅ Retriever built successfully.")

