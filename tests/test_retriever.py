import numpy as np

from retriever import retrieve


class IndexReturning:
    def __init__(self, indices):
        self.indices = indices

    def search(self, query_embedding, top_k):
        self.top_k = top_k
        self.query_embedding = query_embedding
        return np.zeros((1, len(self.indices))), np.array([self.indices])


def test_returns_chunks_in_search_order(fake_model, chunks):
    results = retrieve("question", fake_model, IndexReturning([1, 0]), chunks)

    assert results == [chunks[1], chunks[0]]


def test_passes_top_k_and_float32_query_to_the_index(fake_model, chunks):
    index = IndexReturning([0])

    retrieve("question", fake_model, index, chunks, top_k=3)

    assert index.top_k == 3
    assert index.query_embedding.dtype == np.float32


def test_ignores_empty_faiss_slots(fake_model, chunks):
    # FAISS pads with -1 when the index holds fewer vectors than top_k.
    # Without a check, chunks[-1] would return an unrelated chunk.
    results = retrieve("question", fake_model, IndexReturning([0, -1, -1]), chunks)

    assert results == [chunks[0]]
