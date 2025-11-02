import wikipedia
from langchain.schema import Document

def load_wiki_page(topic: str):
    """Fetch and return a Wikipedia page as a LangChain Document."""
    try:
        summary = wikipedia.summary(topic, auto_suggest=False)
        return [Document(page_content=summary, metadata={"source": topic})]
    except Exception as e:
        print(f"Error loading {topic}: {e}")
        return []
