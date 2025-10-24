def format_sources(docs):
    """Formats retrieved documents for display in Streamlit."""
    if not docs:
        return "No source documents found."
    md = ""
    for d in docs:
        title = d.metadata.get("title", "Unknown")
        snippet = d.page_content[:300].strip().replace("\n", " ")
        md += f"**📘 {title}:** {snippet}...\n\n"
    return md
