"""Stub the modules that need a model download, a FAISS index or an API key.

The tests run without PDFs, network access or a Gemini key, so they can run in CI.
"""
import sys
import types

import pytest

CHUNKS = [
    {"text": "Transformers use self-attention.", "source": "attention.pdf", "page": 3},
    {"text": "RAG retrieves context first.", "source": "rag.pdf", "page": 1},
]


class FakeModel:
    def encode(self, texts):
        return [[0.0, 0.0] for _ in texts]


class FakeIndex:
    """Returns the first two chunks, in the same shape FAISS uses."""

    def search(self, query_embedding, top_k):
        return [[0.1, 0.2]], [[0, 1]]


def _stub(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


_stub("embedder", model=FakeModel())
_stub("vector_store", load_vector_store=lambda: FakeIndex(), load_chunks=lambda: CHUNKS)
_stub("dotenv", load_dotenv=lambda: None)

_genai = _stub("google.genai", Client=lambda api_key=None: types.SimpleNamespace())
_stub("google", genai=_genai)


@pytest.fixture
def chunks():
    return CHUNKS


@pytest.fixture
def fake_model():
    return FakeModel()
