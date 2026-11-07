from embedder import model
from vector_store import ACTIVE_INDEX_PATH, ACTIVE_CHUNKS_PATH, load_vector_store, load_chunks
from retriever import retrieve
from reranker import rerank
from llm import generate_answer, get_sources


# Load previously saved FAISS index and chunks
index = load_vector_store(ACTIVE_INDEX_PATH)
chunks = load_chunks(ACTIVE_CHUNKS_PATH)

print("Saved FAISS index and chunks loaded successfully!")


# Ask a question
query = "What is self-attention?"

# Retrieve relevant chunks
candidates = retrieve(
    query,
    model,
    index,
    chunks,
    top_k=20
)

# Keep the best five
results = rerank(query, candidates, top_k=5)

# Generate answer using Gemini
answer = generate_answer(query, results)

print("\nFINAL ANSWER:")
print(answer)

# Display sources
sources = get_sources(results)

print("\nSOURCES:")
for source in sources:
    print(source)