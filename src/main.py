from fastapi import FastAPI
from pydantic import BaseModel

from embedder import model
from vector_store import load_vector_store, load_chunks
from retriever import retrieve
from llm import generate_answer, get_sources

# Create FastAPI application
app = FastAPI(title="RAG Document Assistant")

# Load saved data
index = load_vector_store()
chunks = load_chunks()

# Request format
class QuestionRequest(BaseModel):
    question: str

# Health check
@app.get("/")
def health_check():
    return {"status": "RAG API is running"}

# Ask a question
@app.post("/ask")
def ask_question(request: QuestionRequest):

    results = retrieve(
        request.question,
        model,
        index,
        chunks,
        top_k=5
    )

    answer = generate_answer(request.question, results)
    sources = get_sources(results)

    return {
        "answer": answer,
        "sources": sources
    }