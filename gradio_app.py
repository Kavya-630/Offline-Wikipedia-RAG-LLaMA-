import os
import gradio as gr
from dotenv import load_dotenv
from retriever import get_retriever
from qa_chain import build_qa_chain

# =====================================
# ⚙️ Load Environment
# =====================================
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMADB_TELEMETRY"] = "OFF"

LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/phi-2.Q4_K_M")
PERSIST_DIR = os.getenv("PERSIST_DIR", "vectorstore")

# =====================================
# 🧠 QA Chain Setup
# =====================================
retriever = get_retriever(k=3)
qa = build_qa_chain(retriever, use_memory=False, use_llama=True)

# =====================================
# 💬 Chat Function
# =====================================
def chat_with_model(message, history):
    """
    Gradio-compatible chat handler.
    Keeps history and appends the latest question/answer pair.
    """
    try:
        print(f"User: {message}")
        result = qa(message)
        answer = result.get("result") or result.get("answer") or "Sorry, I couldn’t generate an answer."
    except Exception as e:
        answer = f"⚠️ Error: {str(e)}"

    history.append((message, answer))
    return history, history


# =====================================
# 🎨 Gradio Chat Interface
# =====================================
with gr.Blocks(theme=gr.themes.Soft(), title="LlamaMind – Wikipedia RAG") as demo:
    gr.Markdown(
        """
        <h1 style='text-align:center; color:#38bdf8;'>🧠 LlamaMind – Wikipedia RAG Chat</h1>
        <p style='text-align:center;'>Chat with a local <b>Phi-2</b> or <b>LLaMA</b> model using Retrieval-Augmented Generation (RAG)</p>
        """
    )

    chatbot = gr.Chatbot(
        height=500,
        show_label=False,
        bubble_full_width=False,
        avatar_images=("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", "https://cdn-icons-png.flaticon.com/512/4712/4712103.png"),
    )

    msg = gr.Textbox(
        placeholder="Type your question about Wikipedia here...",
        label="Ask something",
    )

    clear = gr.Button("🗑️ Clear Chat")

    msg.submit(chat_with_model, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

# =====================================
# 🚀 Launch
# =====================================
if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)
