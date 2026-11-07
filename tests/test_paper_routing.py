import sys
from pathlib import Path

import numpy as np
import pytest

from retriever import PAPER_ALIASES, find_named_papers, retrieve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

from check_retrieval import expected_rank, summarize

ATTENTION = "attention_is_all_you_need.pdf"
CORRECTIVE = "Corrective Retrieval Augmented Generation.pdf"
FEW_SHOT = "Language Models are Few-Shot Learners.pdf"
SURVEY = "Retrieval-Augmented Generation for Lar.pdf"
LEWIS = "Retrieval-Augmented Generation for.pdf"
TRAINING = "Training language models to follow instructions.pdf"


class RankedIndex:
    """Returns chunk positions best-first and cuts them to the requested size."""

    def __init__(self, ranking):
        self.ranking = ranking
        self.requested = None

    def search(self, query_embedding, top_k):
        self.requested = top_k
        found = self.ranking[:top_k]
        return np.zeros((1, len(found))), np.array([found])


def chunk(source, page):
    return {"text": f"{source} {page}", "source": source, "page": page}


@pytest.mark.parametrize(
    "question, paper",
    [
        ("What does Attention Is All You Need propose?", ATTENTION),
        ("Which actions can Corrective RAG trigger?", CORRECTIVE),
        ("How does Corrective Retrieval Augmented Generation work?", CORRECTIVE),
        ("How does CRAG grade documents?", CORRECTIVE),
        ("What does Language Models are Few-Shot Learners show?", FEW_SHOT),
        ("How many parameters does GPT-3 have?", FEW_SHOT),
        ("What does the RAG survey examine?", SURVEY),
        ("What is the difference between RAG-Sequence and RAG-Token?", LEWIS),
        ("What does the original RAG model combine?", LEWIS),
        ("What does Training Language Models to Follow Instructions cover?", TRAINING),
        ("How was InstructGPT trained?", TRAINING),
    ],
)
def test_finds_the_named_paper(question, paper):
    assert find_named_papers(question) == {paper}


@pytest.mark.parametrize(
    "question",
    [
        "What is a Transformer?",
        "How do Transformers use positional encoding?",
        "In the Transformer paper, how many heads are used?",
        "What is the RAG framework?",
        "What is prompt engineering?",
        "How does GPT-3.5 differ from earlier models?",
        "What does the scrag dataset contain?",
    ],
)
def test_generic_terms_name_no_paper(question):
    assert find_named_papers(question) == set()


def test_finds_every_paper_a_question_names():
    question = "How did InstructGPT compare with GPT-3?"

    assert find_named_papers(question) == {TRAINING, FEW_SHOT}


def test_aliases_cover_all_six_papers():
    assert set(PAPER_ALIASES) == {ATTENTION, CORRECTIVE, FEW_SHOT, SURVEY, LEWIS, TRAINING}


def test_no_named_paper_keeps_the_global_search(fake_model):
    chunks = [chunk(CORRECTIVE, 1), chunk(SURVEY, 2), chunk(LEWIS, 3)]
    index = RankedIndex([2, 0, 1])

    results = retrieve("What is the RAG framework?", fake_model, index, chunks, top_k=2)

    assert index.requested == 2
    assert results == [chunks[2], chunks[0]]


def test_named_paper_comes_first_and_keeps_faiss_order(fake_model):
    chunks = [chunk(SURVEY, 1), chunk(CORRECTIVE, 2), chunk(SURVEY, 3), chunk(CORRECTIVE, 4)]
    index = RankedIndex([0, 3, 2, 1])

    results = retrieve("What does Corrective RAG do?", fake_model, index, chunks, top_k=2)

    assert index.requested == len(chunks)
    assert results == [chunks[3], chunks[1]]


def test_other_papers_fill_slots_the_named_paper_cannot(fake_model):
    chunks = [chunk(SURVEY, 1), chunk(CORRECTIVE, 2), chunk(LEWIS, 3)]
    index = RankedIndex([2, 0, 1])

    results = retrieve("What does Corrective RAG do?", fake_model, index, chunks, top_k=2)

    assert results == [chunks[1], chunks[2]]


def test_two_named_papers_share_the_top_slots(fake_model):
    chunks = [chunk(SURVEY, 1), chunk(FEW_SHOT, 2), chunk(TRAINING, 3)]
    index = RankedIndex([0, 2, 1])

    question = "How did InstructGPT compare with GPT-3?"
    results = retrieve(question, fake_model, index, chunks, top_k=2)

    assert results == [chunks[2], chunks[1]]


def test_named_search_ignores_empty_faiss_slots(fake_model):
    chunks = [chunk(SURVEY, 1), chunk(CORRECTIVE, 2)]
    # FAISS pads the end with -1 when it holds fewer vectors than requested.
    index = RankedIndex([1, -1])

    results = retrieve("What does Corrective RAG do?", fake_model, index, chunks)

    assert results == [chunks[1]]


def test_filenames_and_pages_are_unchanged(fake_model):
    chunks = [chunk(SURVEY, 7), chunk(CORRECTIVE, 9)]
    index = RankedIndex([0, 1])

    results = retrieve("What does Corrective RAG do?", fake_model, index, chunks)

    assert [(c["source"], c["page"]) for c in results] == [(CORRECTIVE, 9), (SURVEY, 7)]


def test_expected_rank_needs_the_same_file_and_page():
    retrieved = [chunk(SURVEY, 3), chunk(CORRECTIVE, 2), chunk(CORRECTIVE, 3), chunk(CORRECTIVE, 3)]

    assert expected_rank(retrieved, CORRECTIVE, 3) == 3
    assert expected_rank(retrieved, LEWIS, 3) is None


def test_summarize_counts_hits_and_averages_reciprocal_ranks():
    assert summarize([1, 2, None, 4]) == {"hits": 3, "total": 4, "hit_rate": 0.75, "mrr": 0.4375}
