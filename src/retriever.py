import numpy as np
def retrieve(query, model, index, chunks, top_k=5):

    # Convert the user's question into an embedding
    query_embedding = model.encode([query])

    # FAISS requires float32
    query_embedding = np.array(query_embedding).astype("float32")

    # Search for the closest vectors
    distances, indices = index.search(query_embedding, top_k)

    # Get the corresponding chunks
    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return results