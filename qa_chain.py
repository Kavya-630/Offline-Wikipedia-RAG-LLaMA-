# qa_chain.py
import os
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import LlamaCpp, OpenAI

# ---------------------------------------------------------------------
# LLaMA model path configuration
# ---------------------------------------------------------------------
LLAMA_PATH = os.getenv("LLAMA_MODEL_PATH", "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# Verify model presence
if not os.path.exists(LLAMA_PATH):
    raise FileNotFoundError(
        f"❌ LLaMA model not found at '{LLAMA_PATH}'.\n"
        f"Please download it first using 'download_model.py' or mount your Drive."
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
            temperature=0.2,
            max_tokens=512,
            n_ctx=2048,
            verbose=False,
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
