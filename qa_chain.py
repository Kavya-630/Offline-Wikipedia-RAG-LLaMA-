from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()

def build_qa_chain(retriever, model_path=None, **kwargs):
    if not model_path:
        model_path = os.getenv("LLAMA_MODEL_PATH", "models/phi-2.Q4_K_M")

    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.3,
        max_tokens=512,
        n_ctx=2048,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
    )

    template = """You are a helpful assistant.
Use the context below to answer the question accurately.
If the answer is not in the context, say:
"I don't have enough data to answer that."

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate.from_template(template)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain

