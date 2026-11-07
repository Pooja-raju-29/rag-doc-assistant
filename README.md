# RAG Document Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from technical research papers and PDF documents.

The application combines semantic search using FAISS with Google's Gemini LLM to generate context-aware answers and provide document-level source references.

## Project Overview

Large Language Models (LLMs) may generate incorrect information or lack access to specific documents. Retrieval-Augmented Generation (RAG) addresses this problem by retrieving relevant information from an external knowledge base and supplying it to an LLM before generating an answer.

This project implements an end-to-end RAG pipeline that allows users to ask questions about machine learning, transformers, LLMs, and related technical topics.

## Architecture

**PDF Documents → Text Extraction → Token-Based Chunking → Embeddings → FAISS Vector Store → Document-Aware Retrieval → Cross-Encoder Reranking → Gemini LLM → Answer + Sources**

The system operates in two stages:

**1. Document Processing and Indexing**

- Extract text from PDF documents using PyMuPDF.
- Split extracted text into overlapping chunks of up to 224 tokens, with 32 tokens of overlap, so each chunk fits the embedding model's 256-token limit.
- Generate vector embeddings using Sentence Transformers.
- Store embeddings in a FAISS vector index.
- Save the index and chunk metadata locally for reuse.

**2. Question Answering**

- Accept a user's question through a FastAPI endpoint.
- Convert the question into an embedding.
- Retrieve the top 20 candidate chunks using FAISS. If the question names a known paper, that paper's chunks are put first.
- Rerank the candidates with a local cross-encoder and keep the best five.
- Pass the retrieved context and question to Gemini.
- Return a generated answer along with the retrieved document sources and page numbers.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Text embedding generation |
| all-MiniLM-L6-v2 | Pretrained embedding model |
| FAISS | Vector similarity search |
| ms-marco-MiniLM-L6-v2 | Local cross-encoder reranking |
| Google Gemini API | LLM-based answer generation |
| FastAPI | REST API development |
| Pydantic | Request validation |
| Docker | Application containerization |
| Streamlit | Chat interface |
| GitHub Actions | CI: tests, syntax check and Docker build |
| Pytest | Automated tests |
| Git & GitHub | Version control |

## Key Features

- Multi-document PDF question answering
- Semantic search using vector embeddings
- Overlapping text chunking for context preservation
- Document-aware retrieval and cross-encoder reranking
- LLM-generated answers grounded in retrieved context
- Source document and page-number references
- Persistent FAISS vector index
- REST API built with FastAPI
- Docker-based application execution
- Streamlit chat interface
- Automated tests for chunking, retrieval, sources and the API

## Dataset and Knowledge Base

The knowledge base consists of technical research papers covering topics such as:

- Transformer architecture and self-attention
- Retrieval-Augmented Generation
- Large Language Models
- Instruction tuning and language model capabilities

PDF documents are stored locally in the `data/` directory and are not included in the GitHub repository.

## API Endpoint

### POST `/ask`

Accepts a question and returns an answer with supporting document references.

**Example request**

```json
{
  "question": "What is the transformer architecture?"
}
```

**Example response**

```json
{
  "answer": "The Transformer is a neural network architecture based on attention mechanisms...",
  "sources": [
    "attention_is_all_you_need.pdf - Page 3",
    "attention_is_all_you_need.pdf - Page 5"
  ]
}
```

*The response shown above is illustrative; actual answers and retrieved pages depend on the indexed documents.*

### Error responses

- `422`: the question is missing, empty or blank.
- `503`: the LLM call failed.

## Running the Application

### Prerequisites

- Python 3.11
- Docker (optional)
- Google Gemini API key
- PDF documents for indexing
- A generated FAISS index and chunk metadata

### Environment Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Never commit your API key to GitHub.

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Build the Index

Put your PDF documents in `data/`, then run this from the project root:

```bash
python src/ingest.py
```

### Start the FastAPI Application

Once the FAISS index and chunk metadata have been generated, run the command below. The first start downloads the embedding and reranking models from Hugging Face, for free, and they then run locally.

```bash
uvicorn main:app --app-dir src --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

Use the `/ask` endpoint to submit a question.

### Open the Chat Interface

With the FastAPI application running on port 8000, open a second terminal:

```bash
pip install -r requirements-ui.txt
streamlit run chat_ui.py
```

### Run with Docker

Build the Docker image:

```bash
docker build -t rag-doc-assistant .
```

Run the container:

```bash
docker run -d --name rag-container -p 8000:8000 --env-file .env -v "${PWD}/storage:/app/storage:ro" rag-doc-assistant
```

Open:

http://localhost:8000/docs

**Note:** The image does not contain the models, so a new container downloads them when it starts. It also does not contain the FAISS index or chunk metadata, because `storage/` is not in Git and CI builds the image from Git. Build the index first, then mount `storage/` when starting the container as shown above. The API cannot start without it.

## Running the Tests

The tests need no PDFs, model download or Gemini API key, because the heavy modules are stubbed:

```bash
pip install -r requirements-dev.txt
pytest
```

GitHub Actions runs them on pushes and pull requests to `main`.

## Retrieval Evaluation

Retrieval was compared offline on 24 verified questions across the 6 papers, with no LLM calls. Every setup uses document-aware search and keeps 5 chunks.

| Setup | Expected-page Hit@5 | Expected-page MRR | Full evidence retrieved | Average context tokens |
|---|---|---|---|---|
| 540 word-based chunks | 20/24 | 0.612 | 13/24 | 1,934 |
| 1,011 token-based chunks | 18/24 | 0.560 | 11/24 | 1,013 |
| 540 word-based chunks + reranking | 22/24 | 0.722 | 18/24 | 2,071 |
| **1,011 token-based chunks + reranking (active)** | 20/24 | 0.729 | 19/24 | 1,058 |

To reproduce it, with the original index backed up in `storage/baseline_540/`:

```bash
.venv/Scripts/python.exe evals/compare_retrieval.py
```

These numbers measure retrieval only, on a small sample. The Ragas scores in `evals/RAG_Evaluation_Report.md` come from the earlier global search on the 540-chunk index, and answer quality has not been re-measured for the active setup.

## Project Outcomes

This project demonstrates practical experience with:

- Designing and implementing end-to-end RAG pipelines
- Working with pretrained embedding models
- Building vector similarity search systems
- Integrating external LLM APIs
- Developing REST APIs for AI applications
- Managing persistent vector indexes
- Containerizing AI applications using Docker

## Future Improvements

- Automated document ingestion and index rebuilding
- Re-measuring answer quality on the current retrieval setup
- Improved citation accuracy
- Cloud deployment

## Author

Developed as a hands-on AI engineering portfolio project focused on Retrieval-Augmented Generation, LLM integration, API development, and containerization.
