# import faiss
# import numpy as np

# def create_vector_store(embeddings):

#     embeddings = np.array(embeddings).astype("float32")

#     dimension = embeddings.shape[1]

#     index = faiss.IndexFlatL2(dimension)

#     index.add(embeddings)

#     return index

import faiss
import numpy as np
import pickle
def create_vector_store(embeddings):
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index
def save_vector_store(index, path="storage/faiss_index.bin"):
    faiss.write_index(index, path)
def load_vector_store(path="storage/faiss_index.bin"):
    index = faiss.read_index(path)
    return index

def save_chunks(chunks, path="storage/chunks.pkl"):
    with open(path, "wb") as file:
        pickle.dump(chunks, file)


def load_chunks(path="storage/chunks.pkl"):
    with open(path, "rb") as file:
        chunks = pickle.load(file)

    return chunks