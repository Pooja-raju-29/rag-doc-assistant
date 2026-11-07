import pytest

import reranker
from reranker import rerank


class ScoringModel:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs, show_progress_bar=True):
        self.pairs = pairs
        return self.scores


def chunk(name, page=1):
    return {"text": f"text {name}", "source": f"{name}.pdf", "page": page}


@pytest.fixture
def use_scores(monkeypatch):
    def apply(scores):
        scoring = ScoringModel(scores)
        monkeypatch.setattr(reranker, "model", scoring)
        return scoring

    return apply


def test_highest_scores_come_first(use_scores):
    chunks = [chunk("a"), chunk("b"), chunk("c")]
    use_scores([0.1, 0.9, 0.5])

    assert rerank("question", chunks) == [chunks[1], chunks[2], chunks[0]]


def test_keeps_only_top_k(use_scores):
    chunks = [chunk("a"), chunk("b"), chunk("c")]
    use_scores([0.1, 0.9, 0.5])

    assert rerank("question", chunks, top_k=2) == [chunks[1], chunks[2]]


def test_negative_scores_are_ordered_correctly(use_scores):
    chunks = [chunk("a"), chunk("b")]
    use_scores([-5.0, -1.0])

    assert rerank("question", chunks) == [chunks[1], chunks[0]]


def test_equal_scores_keep_search_order(use_scores):
    chunks = [chunk("a"), chunk("b"), chunk("c")]
    use_scores([0.5, 0.5, 0.5])

    assert rerank("question", chunks) == chunks


def test_scores_each_chunk_with_the_question(use_scores):
    chunks = [chunk("a"), chunk("b")]
    scoring = use_scores([0.1, 0.2])

    rerank("question", chunks)

    assert scoring.pairs == [("question", "text a"), ("question", "text b")]


def test_filenames_and_pages_are_unchanged(use_scores):
    chunks = [chunk("a", page=4), chunk("b", page=9)]
    use_scores([0.1, 0.2])

    results = rerank("question", chunks)

    assert [(c["source"], c["page"]) for c in results] == [("b.pdf", 9), ("a.pdf", 4)]


def test_no_chunks_gives_no_results(use_scores):
    scoring = use_scores([])

    assert rerank("question", []) == []
    assert scoring.pairs is None
