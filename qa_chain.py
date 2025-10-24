# qa_chain.py
import os
import gdown
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import LlamaCpp, OpenAI

# Paths and Keys
LLAMA_PATH = os.environ.get("LLAMA_MODEL_PATH", "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# ✅ Check if model exists, else download from Drive
if not os.path.exists(LLAMA_PATH):
    os.makedirs(os.path.dirname(LLAMA_PATH), exist_ok=True)
    print("[INFO] Model file not found. Downloading from Google Drive...")
    url = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # ⬅️ paste your Drive file ID here
    gdown.download(url, LLAMA_PATH, quiet=False)
    print("[INFO] Download complete!")

# Rest of your code
def build_qa_chain(retriever, use_memory: bool = True, use_llama: bool = False):
    if use_llama:
        print(f"[INFO] Using local LLaMA model from {LLAMA_PATH}")
        llm = LlamaCpp(
            model_path=LLAMA_PATH,
            temperature=0.2,
            max_tokens=512,
            n_ctx=2048,
            verbose=False,
            n_threads=4,
        )
    else:
        OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
        llm = OpenAI(openai_api_key=OPENAI_KEY, temperature=0.0)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) if use_memory else None

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa
