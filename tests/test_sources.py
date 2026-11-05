from llm import get_sources


def test_formats_source_and_page(chunks):
    assert get_sources(chunks[:1]) == ["attention.pdf - Page 3"]


def test_removes_duplicates_and_keeps_order():
    retrieved = [
        {"source": "b.pdf", "page": 2},
        {"source": "a.pdf", "page": 1},
        {"source": "b.pdf", "page": 2},
    ]

    assert get_sources(retrieved) == ["b.pdf - Page 2", "a.pdf - Page 1"]


def test_same_file_on_different_pages_stays_separate():
    retrieved = [{"source": "a.pdf", "page": 1}, {"source": "a.pdf", "page": 2}]

    assert get_sources(retrieved) == ["a.pdf - Page 1", "a.pdf - Page 2"]


def test_no_chunks_gives_no_sources():
    assert get_sources([]) == []
