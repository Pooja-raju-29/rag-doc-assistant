from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints

from embedder import model
from vector_store import ACTIVE_INDEX_PATH, ACTIVE_CHUNKS_PATH, load_vector_store, load_chunks
from retriever import retrieve
from reranker import rerank
from llm import generate_answer, get_sources

# Create FastAPI application
app = FastAPI(title="RAG Document Assistant")

# Load saved data
index = load_vector_store(ACTIVE_INDEX_PATH)
chunks = load_chunks(ACTIVE_CHUNKS_PATH)

# Search finds this many chunks, then the reranker keeps the best five
CANDIDATE_COUNT = 20

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

    candidates = retrieve(
        request.question,
        model,
        index,
        chunks,
        top_k=CANDIDATE_COUNT
    )

    results = rerank(request.question, candidates, top_k=5)

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