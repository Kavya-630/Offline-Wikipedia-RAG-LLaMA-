from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()
def build_or_load_vectorstore(docs, chunk_size=500, chunk_overlap=50):
    """
    Builds a Chroma vectorstore from documents and saves it.
    If already exists, loads it from disk.
    """
    persist_dir = os.getenv("PERSIST_DIR", "vectorstore")
    embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    embedding_function = SentenceTransformerEmbeddings(model_name=embed_model)

    # Check if vectorstore already exists
    if os.path.exists(os.path.join(persist_dir, "chroma-collections.parquet")):
        print(f"🔄 Loading existing Chroma store from {persist_dir}...")
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_function
        )
    else:
        print(f"🧠 Building new Chroma store at {persist_dir}...")
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding_function,
            persist_directory=persist_dir
        )
        vectordb.persist()

    return vectordb

def create_retriever(docs):
    persist_dir = os.getenv("PERSIST_DIR", "vectorstore")
    embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    embedding_function = SentenceTransformerEmbeddings(model_name=embed_model)

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding_function,
        persist_directory=persist_dir
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return retriever

