def create_chunks(documents, chunk_size=300, overlap=50):
    chunks = []
    for document in documents:
        text = document["text"]
        words = text.split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunk_data = {
                "text": chunk_text,
                "source": document["source"],
                "page": document["page"]
            }
            chunks.append(chunk_data)
            start = end - overlap
    return chunks