from embedder import model
from vector_store import load_vector_store, load_chunks
from retriever import retrieve
from llm import generate_answer, get_sources


# Load previously saved FAISS index and chunks
index = load_vector_store()
chunks = load_chunks()

print("Saved FAISS index and chunks loaded successfully!")


# Ask a question
query = "What is self-attention?"

# Retrieve relevant chunks
results = retrieve(
    query,
    model,
    index,
    chunks,
    top_k=5
)

# Generate answer using Gemini
answer = generate_answer(query, results)

print("\nFINAL ANSWER:")
print(answer)

# Display sources
sources = get_sources(results)

print("\nSOURCES:")
for source in sources:
    print(source)