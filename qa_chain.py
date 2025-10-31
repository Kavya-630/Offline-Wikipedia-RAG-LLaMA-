import os
import gdown
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import LlamaCpp, OpenAI

# ---------------------------------------------------------------------
# LLaMA model path configuration
# ---------------------------------------------------------------------
LLAMA_PATH = os.getenv("LLAMA_MODEL_PATH", "models/tinyllama-1.1b-chat.Q4_K_M.gguf")
LLAMA_URL = "https://drive.google.com/uc?export=download&id=1Yy3ItSh6tFbGN75_Vd00kX068GN9jESz"
# ✅ Auto-download model if missing
os.makedirs(os.path.dirname(LLAMA_PATH), exist_ok=True)
if not os.path.exists(LLAMA_PATH):
    print(f"📥 Model not found locally. Downloading from {LLAMA_URL} ...")
    try:
        gdown.download(LLAMA_URL, LLAMA_PATH, quiet=False)
        print(f"✅ Model downloaded successfully to {LLAMA_PATH}")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        raise FileNotFoundError(
            f"Could not download LLaMA model. Please ensure the link is correct: {LLAMA_URL}"
        )

# ---------------------------------------------------------------------
# QA chain builder
# ---------------------------------------------------------------------
def build_qa_chain(retriever, use_memory: bool = True, use_llama: bool = False):
    """
    Build a RetrievalQA chain with either local LLaMA or OpenAI.
    """
    if use_llama:
        print(f"[INFO] Using local LLaMA model from {LLAMA_PATH}")
        llm = LlamaCpp(
            model_path=LLAMA_PATH,
            temperature=0.7,
            max_new_tokens=512,
            n_ctx=4096,
            verbose=False,
            n_batch=512,
            n_threads=4,
        )

    else:
        OPENAI_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_KEY:
            raise ValueError("❌ OPENAI_API_KEY not found in environment variables.")
        llm = OpenAI(openai_api_key=OPENAI_KEY, temperature=0.0)

    # Optional memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) if use_memory else None

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        memory=memory,
    )
    return qa




