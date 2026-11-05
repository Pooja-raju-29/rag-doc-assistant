from chunker import create_chunks


def document(word_count):
    return {
        "text": " ".join(f"word{i}" for i in range(word_count)),
        "source": "paper.pdf",
        "page": 7,
    }


def test_short_document_gives_one_chunk():
    chunks = create_chunks([document(10)], chunk_size=300, overlap=50)

    assert len(chunks) == 1
    assert chunks[0]["text"].split()[0] == "word0"


def test_consecutive_chunks_overlap():
    chunks = create_chunks([document(30)], chunk_size=20, overlap=5)

    first = chunks[0]["text"].split()
    second = chunks[1]["text"].split()
    assert first[-5:] == second[:5]


def test_no_chunk_is_longer_than_chunk_size():
    chunks = create_chunks([document(100)], chunk_size=20, overlap=5)

    assert all(len(chunk["text"].split()) <= 20 for chunk in chunks)


def test_every_word_is_kept():
    chunks = create_chunks([document(100)], chunk_size=20, overlap=5)

    kept = {word for chunk in chunks for word in chunk["text"].split()}
    assert kept == {f"word{i}" for i in range(100)}


def test_chunks_keep_source_and_page():
    chunks = create_chunks([document(30)], chunk_size=20, overlap=5)

    assert all(chunk["source"] == "paper.pdf" for chunk in chunks)
    assert all(chunk["page"] == 7 for chunk in chunks)


def test_empty_document_list_gives_no_chunks():
    assert create_chunks([]) == []
