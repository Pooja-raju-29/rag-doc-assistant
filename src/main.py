from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints

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
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

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

    try:
        answer = generate_answer(request.question, results)
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="The answer service is unavailable."
        ) from error

    sources = get_sources(results)

    return {
        "answer": answer,
        "sources": sources
    }