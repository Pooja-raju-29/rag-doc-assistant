# RAG Document Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from technical research papers and PDF documents.

The application combines semantic search using FAISS with Google's Gemini LLM to generate context-aware answers and provide document-level source references.

## Project Overview

Large Language Models (LLMs) may generate incorrect information or lack access to specific documents. Retrieval-Augmented Generation (RAG) addresses this problem by retrieving relevant information from an external knowledge base and supplying it to an LLM before generating an answer.

This project implements an end-to-end RAG pipeline that allows users to ask questions about machine learning, transformers, LLMs, and related technical topics.

## Architecture

**PDF Documents → Text Extraction → Chunking → Embeddings → FAISS Vector Store → Semantic Retrieval → Gemini LLM → Answer + Sources**

The system operates in two stages:

**1. Document Processing and Indexing**

- Extract text from PDF documents using PyMuPDF.
- Split extracted text into smaller overlapping chunks.
- Generate vector embeddings using Sentence Transformers.
- Store embeddings in a FAISS vector index.
- Save the index and chunk metadata locally for reuse.

**2. Question Answering**

- Accept a user's question through a FastAPI endpoint.
- Convert the question into an embedding.
- Retrieve the top five relevant document chunks using FAISS.
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
| Google Gemini API | LLM-based answer generation |
| FastAPI | REST API development |
| Pydantic | Request validation |
| Docker | Application containerization |
| Git & GitHub | Version control |

## Key Features

- Multi-document PDF question answering
- Semantic search using vector embeddings
- Overlapping text chunking for context preservation
- Top-K document retrieval
- LLM-generated answers grounded in retrieved context
- Source document and page-number references
- Persistent FAISS vector index
- REST API built with FastAPI
- Docker-based application execution

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

### Start the FastAPI Application

Once the FAISS index and chunk metadata have been generated, run:

```bash
uvicorn main:app --app-dir src --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

Use the `/ask` endpoint to submit a question.

### Run with Docker

Build the Docker image:

```bash
docker build -t rag-doc-assistant .
```

Run the container:

```bash
docker run -d --name rag-container -p 8000:8000 --env-file .env rag-doc-assistant
```

Open:

http://localhost:8000/docs

**Note:** The Docker image currently expects the saved FAISS index and chunk metadata to exist in `storage/` before building.

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
- Retrieval quality evaluation
- Improved citation accuracy
- Automated testing and GitHub Actions CI/CD
- Cloud deployment
- Web-based user interface

## Author

Developed as a hands-on AI engineering portfolio project focused on Retrieval-Augmented Generation, LLM integration, API development, and containerization.
