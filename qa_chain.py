# qa_chain.py
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub, OpenAI
from langchain_community.llms import Ollama

def build_qa_chain(retriever, model_type="huggingface", model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Build a Retrieval-QA chain that connects an LLM with the document retriever.
    """

    # Select model backend
    if model_type == "huggingface":
        llm = HuggingFaceHub(
            repo_id=model_name,
            model_kwargs={"temperature": 0.3, "max_new_tokens": 512}
        )
    elif model_type == "openai":
        llm = OpenAI(model_name=model_name)
    elif model_type == "ollama":
        llm = Ollama(model=model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Prompt
    template = """You are a helpful AI assistant. Use the provided context to answer the question accurately.
If the context does not contain the answer, say so.

Context:
{context}

Question:
{question}

Answer:"""

    qa_prompt = PromptTemplate.from_template(template)

    # Create QA Chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
    )

    return chain
