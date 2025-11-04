from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

def build_or_load_vectorstore(docs=None):
    """Create or load Chroma vectorstore."""
    persist_dir = os.getenv("PERSIST_DIR", "vectorstore")
    embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    embedding_function = SentenceTransformerEmbeddings(model_name=embed_model)

    # If documents are provided, build and persist
    if docs:
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding_function,
            persist_directory=persist_dir
        )
        vectordb.persist()
    else:
        # Otherwise, load existing
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_function
        )

    return vectordb


def create_retriever(k: int = 3):
    """Load persisted vectorstore and return retriever."""
    persist_dir = os.getenv("PERSIST_DIR", "vectorstore")
    embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    embedding_function = SentenceTransformerEmbeddings(model_name=embed_model)

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_function
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever
