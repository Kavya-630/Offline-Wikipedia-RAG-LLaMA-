import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document  # ✅ updated import

PERSIST_DIR = os.environ.get("PERSIST_DIR", "vectorstore")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

def build_or_load_vectorstore(docs: List[Document],
                              persist_directory: str = PERSIST_DIR,
                              embedding_model: str = EMBED_MODEL,
                              chunk_size: int = 500,
                              chunk_overlap: int = 50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    embedder = SentenceTransformerEmbeddings(model_name=embedding_model)
    vectordb = Chroma.from_documents(chunks, embedder, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb

def get_retriever(k: int = 3, persist_directory: str = PERSIST_DIR):
    embedder = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedder)
    return vectordb.as_retriever(search_kwargs={"k": k})
