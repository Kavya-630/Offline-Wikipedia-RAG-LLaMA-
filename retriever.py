from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

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
