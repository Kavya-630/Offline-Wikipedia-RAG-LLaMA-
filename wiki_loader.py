import wikipedia
from langchain.schema import Document

def fetch_wikipedia_pages(topics, max_pages_per_topic=2):
    """Fetch and clean Wikipedia pages for given topics."""
    docs = []
    for topic in topics:
        try:
            results = wikipedia.search(topic, results=max_pages_per_topic)
            for title in results:
                try:
                    content = wikipedia.page(title).content
                    docs.append(Document(page_content=content, metadata={"title": title}))
                    print(f"[INFO] Fetched: {title}")
                except Exception as e:
                    print(f"[WARN] Could not fetch {title}: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to search for {topic}: {e}")
    print(f"[INFO] Total pages fetched: {len(docs)}")
    return docs
