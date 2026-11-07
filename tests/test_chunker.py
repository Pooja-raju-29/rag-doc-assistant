import re
import sys

import pytest

from chunker import create_chunks

SPECIAL_TOKENS = 2


class FakeTokenizer:
    """Treats each word as one token and adds two special tokens, like the real one."""

    def num_special_tokens_to_add(self, pair=False):
        return SPECIAL_TOKENS

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, verbose=True):
        spans = [match.span() for match in re.finditer(r"\S+", text)]
        encoded = {"input_ids": list(range(len(spans) + (SPECIAL_TOKENS if add_special_tokens else 0)))}
        if return_offsets_mapping:
            encoded["offset_mapping"] = spans
        return encoded


class GrowingTokenizer(FakeTokenizer):
    """Reports one extra token when a chunk is checked, like a chunk starting mid-word."""

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, verbose=True):
        encoded = super().__call__(text, add_special_tokens, return_offsets_mapping, verbose)
        if add_special_tokens:
            encoded["input_ids"].append(0)
        return encoded


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


def document(word_count, source="paper.pdf", page=7):
    return {
        "text": " ".join(f"word{i}" for i in range(word_count)),
        "source": source,
        "page": page,
    }


def test_short_document_gives_one_chunk(tokenizer):
    chunks = create_chunks([document(10)], tokenizer=tokenizer)

    assert len(chunks) == 1
    assert chunks[0]["text"].split()[0] == "word0"


def test_consecutive_chunks_share_exactly_the_overlap(tokenizer):
    chunks = create_chunks([document(30)], chunk_size=20, overlap=5, tokenizer=tokenizer)

    first = chunks[0]["text"].split()
    second = chunks[1]["text"].split()
    assert first[-5:] == second[:5]
    assert first[-6] not in second


def test_chunk_size_includes_the_special_tokens(tokenizer):
    chunks = create_chunks([document(100)], chunk_size=20, overlap=5, tokenizer=tokenizer)

    assert len(chunks[0]["text"].split()) == 20 - SPECIAL_TOKENS
    assert all(len(tokenizer(chunk["text"])["input_ids"]) <= 20 for chunk in chunks)


def test_default_size_is_224_tokens_with_32_overlap(tokenizer):
    chunks = create_chunks([document(1000)], tokenizer=tokenizer)

    first = chunks[0]["text"].split()
    second = chunks[1]["text"].split()
    assert len(first) == 224 - SPECIAL_TOKENS
    assert first[-32:] == second[:32]
    assert first[-33] not in second


def test_default_tokenizer_comes_from_the_embedding_model(monkeypatch, tokenizer):
    monkeypatch.setattr(sys.modules["embedder"].model, "tokenizer", tokenizer, raising=False)

    chunks = create_chunks([document(10)])

    assert len(chunks) == 1


def test_every_word_is_kept(tokenizer):
    chunks = create_chunks([document(100)], chunk_size=20, overlap=5, tokenizer=tokenizer)

    kept = {word for chunk in chunks for word in chunk["text"].split()}
    assert kept == {f"word{i}" for i in range(100)}


def test_no_extra_chunk_after_the_text_has_ended(tokenizer):
    chunks = create_chunks([document(18)], chunk_size=20, overlap=5, tokenizer=tokenizer)

    assert len(chunks) == 1


def test_chunk_that_grows_is_trimmed_and_keeps_the_overlap():
    tokenizer = GrowingTokenizer()

    chunks = create_chunks([document(60)], chunk_size=20, overlap=5, tokenizer=tokenizer)

    first = chunks[0]["text"].split()
    second = chunks[1]["text"].split()
    assert len(first) == 17
    assert first[-5:] == second[:5]
    assert all(len(tokenizer(chunk["text"])["input_ids"]) <= 20 for chunk in chunks)


def test_chunks_keep_source_and_page(tokenizer):
    pages = [document(30, "a.pdf", 1), document(30, "b.pdf", 9)]

    chunks = create_chunks(pages, chunk_size=20, overlap=5, tokenizer=tokenizer)

    assert {(chunk["source"], chunk["page"]) for chunk in chunks} == {("a.pdf", 1), ("b.pdf", 9)}
    assert all(set(chunk) == {"text", "source", "page"} for chunk in chunks)


def test_chunks_never_cross_pages(tokenizer):
    pages = [document(30, page=1), {"text": "other text", "source": "paper.pdf", "page": 2}]

    chunks = create_chunks(pages, chunk_size=20, overlap=5, tokenizer=tokenizer)

    assert [chunk["text"] for chunk in chunks if chunk["page"] == 2] == ["other text"]


def test_blank_page_gives_no_chunks(tokenizer):
    blank = {"text": " \n ", "source": "paper.pdf", "page": 1}

    assert create_chunks([blank], tokenizer=tokenizer) == []


def test_empty_document_list_gives_no_chunks():
    assert create_chunks([]) == []


@pytest.mark.parametrize("overlap", [18, 50, -1])
def test_overlap_that_cannot_fit_is_rejected(tokenizer, overlap):
    with pytest.raises(ValueError):
        create_chunks([document(30)], chunk_size=20, overlap=overlap, tokenizer=tokenizer)


@pytest.fixture
def real_tokenizer():
    # The light CI install has no transformers or model, so these tests skip there
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True
        )
    except OSError:
        pytest.skip("all-MiniLM-L6-v2 is not on this computer")


def page_of(text):
    return {"text": text, "source": "paper.pdf", "page": 3}


def test_real_tokenizer_limit_and_overlap(real_tokenizer):
    # Common words are one token each, so the overlap can be compared exactly
    words = "the quick brown fox jumps over a lazy dog and then runs home ".split()
    chunks = create_chunks([page_of(" ".join(words * 100))], tokenizer=real_tokenizer)

    ids = [real_tokenizer(chunk["text"])["input_ids"] for chunk in chunks]
    assert len(chunks) > 1
    assert len(ids[0]) == 224
    assert all(len(chunk_ids) <= 224 for chunk_ids in ids)
    assert ids[0][1:-1][-32:] == ids[1][1:-1][:32]


def test_real_tokenizer_limit_holds_for_words_split_across_chunks(real_tokenizer):
    # Long rare words become many tokens, so chunk boundaries land inside them
    text = "Supercalifragilistic Tokenization electroencephalography, " * 300
    chunks = create_chunks([page_of(text)], tokenizer=real_tokenizer)

    assert all(len(real_tokenizer(chunk["text"])["input_ids"]) <= 224 for chunk in chunks)


def test_real_tokenizer_keeps_the_original_text(real_tokenizer):
    text = "RAG-Token uses BERT, not GPT-3."

    chunks = create_chunks([page_of(text)], tokenizer=real_tokenizer)

    assert chunks[0]["text"] == text
