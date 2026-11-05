import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "generate_answer", lambda question, results: "An answer.")
    return TestClient(main.app)


def test_health_check(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "RAG API is running"}


def test_ask_returns_answer_and_sources(client):
    response = client.post("/ask", json={"question": "What is attention?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "An answer.",
        "sources": ["attention.pdf - Page 3", "rag.pdf - Page 1"],
    }


def test_ask_passes_the_question_and_retrieved_chunks_to_the_llm(monkeypatch, chunks):
    calls = []

    def fake_generate(question, results):
        calls.append((question, results))
        return "An answer."

    monkeypatch.setattr(main, "generate_answer", fake_generate)

    TestClient(main.app).post("/ask", json={"question": "What is RAG?"})

    assert calls == [("What is RAG?", chunks)]


@pytest.mark.parametrize(
    "body",
    [{}, {"question": 123}, {"question": ""}, {"question": "   "}],
)
def test_ask_rejects_invalid_questions(client, body):
    assert client.post("/ask", json=body).status_code == 422


def test_ask_returns_503_when_the_llm_fails(monkeypatch):
    def failing_generate(question, results):
        raise RuntimeError("Gemini is unavailable")

    monkeypatch.setattr(main, "generate_answer", failing_generate)

    response = TestClient(main.app).post("/ask", json={"question": "What is RAG?"})

    assert response.status_code == 503
    assert response.json() == {"detail": "The answer service is unavailable."}
