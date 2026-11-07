import re

import numpy as np

# Titles and aliases that identify one paper. Generic words such as
# "Transformer" and plain "RAG" are left out because they fit several papers.
PAPER_ALIASES = {
    "attention_is_all_you_need.pdf": [r"attention is all you need"],
    "Corrective Retrieval Augmented Generation.pdf": [r"corrective (rag|retrieval)", r"\bcrag\b"],
    "Language Models are Few-Shot Learners.pdf": [r"few-shot learners", r"\bgpt-3\b(?!\.\d)"],
    "Retrieval-Augmented Generation for Lar.pdf": [r"\brag survey\b"],
    "Retrieval-Augmented Generation for.pdf": [r"\brag-(sequence|token)\b", r"\boriginal rag\b"],
    "Training language models to follow instructions.pdf": [r"training language models", r"\binstructgpt\b"],
}


def find_named_papers(query):
    query = query.lower()
    return {
        source for source, patterns in PAPER_ALIASES.items()
        if any(re.search(pattern, query) for pattern in patterns)
    }


def retrieve(query, model, index, chunks, top_k=5):

    # Convert the user's question into an embedding
    query_embedding = model.encode([query])

    # FAISS requires float32
    query_embedding = np.array(query_embedding).astype("float32")

    named_papers = find_named_papers(query)

    # Rank every chunk when a paper is named, so its chunks can be moved first
    search_size = len(chunks) if named_papers else top_k

    # Search for the closest vectors
    distances, indices = index.search(query_embedding, search_size)

    # Get the corresponding chunks
    results = []

    for i in indices[0]:
        # FAISS returns -1 when it has fewer vectors than top_k
        if i < 0:
            continue
        results.append(chunks[i])

    # Named papers first; the sort keeps FAISS order within each group
    results.sort(key=lambda chunk: chunk["source"] not in named_papers)

    return results[:top_k]
