from sentence_transformers import CrossEncoder

# Load pretrained reranking model
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


def rerank(query, chunks, top_k=5):

    if not chunks:
        return []

    # Score each chunk by reading it together with the question
    scores = model.predict(
        [(query, chunk["text"]) for chunk in chunks],
        show_progress_bar=False
    )

    # Highest score first; ties keep their search order
    order = sorted(range(len(chunks)), key=lambda i: -float(scores[i]))

    return [chunks[i] for i in order[:top_k]]
