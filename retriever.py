# retriever.py
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_retriever(data_dir="data", index_path="faiss_index"):
    """
    Build or load a FAISS retriever from local documents.
    """
    # Load all .txt files
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build or load FAISS index
    if os.path.exists(index_path):
        print("📦 Loading existing FAISS index...")
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("🧩 Building new FAISS index...")
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(index_path)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever
