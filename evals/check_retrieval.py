"""Compare global search with document-aware search on the 24 evaluation questions.

Runs on this computer only: no Gemini calls, and any outside connection is blocked.
It prints its results and writes no files.

Run from anywhere with the main environment:
    .venv/Scripts/python.exe evals/check_retrieval.py

Exit code 1 means a pass criterion failed.
"""
import json
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP_K = 5


def expected_rank(retrieved, source, page):
    """Position (from 1) of the first chunk on the expected page, or None."""
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk["source"] == source and chunk["page"] == page:
            return rank
    return None


def summarize(ranks):
    hits = sum(rank is not None for rank in ranks)
    mrr = sum(1 / rank for rank in ranks if rank is not None) / len(ranks)
    return {"hits": hits, "total": len(ranks), "hit_rate": hits / len(ranks), "mrr": mrr}


def block_outside_network():
    def blocked(*args, **kwargs):
        raise OSError("check_retrieval.py does not use the network")

    socket.getaddrinfo = blocked
    socket.socket.connect = blocked


def global_search(question, model, index, chunks):
    """The search as it worked before papers were recognised."""
    query_embedding = model.encode([question]).astype("float32")
    _, indices = index.search(query_embedding, TOP_K)
    return [chunks[i] for i in indices[0] if i >= 0]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    block_outside_network()

    # The saved index and chunks use paths relative to the project folder
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))

    from embedder import model
    from retriever import find_named_papers, retrieve
    from vector_store import load_chunks, load_vector_store

    index = load_vector_store()
    chunks = load_chunks()
    questions = json.loads((ROOT / "evals" / "eval_dataset.json").read_text(encoding="utf-8"))

    before_ranks, after_ranks = [], []
    right_pdf_before = right_pdf_after = 0
    failures = []

    print(f"{'ID':>3}  {'Paper named':<12} {'Rank before':>11} {'Rank after':>10}  Correct-PDF chunks")
    for item in questions:
        question, source, page = item["question"], item["source"], item["page"]
        before = global_search(question, model, index, chunks)
        after = retrieve(question, model, index, chunks, top_k=TOP_K)
        named = find_named_papers(question)

        rank_before = expected_rank(before, source, page)
        rank_after = expected_rank(after, source, page)
        before_ranks.append(rank_before)
        after_ranks.append(rank_after)

        count_before = sum(chunk["source"] == source for chunk in before)
        count_after = sum(chunk["source"] == source for chunk in after)
        right_pdf_before += count_before
        right_pdf_after += count_after

        if rank_before is not None and rank_after is None:
            failures.append(f"Q{item['id']} lost its expected page")
        if not named and after != before:
            failures.append(f"Q{item['id']} names no paper but its results changed")
        if any(not chunk.get("source") or not chunk.get("page") for chunk in after):
            failures.append(f"Q{item['id']} returned a chunk without a filename or page")

        label = "two papers" if len(named) > 1 else "yes" if named else "no"
        print(
            f"{item['id']:>3}  {label:<12} {str(rank_before or '-'):>11} {str(rank_after or '-'):>10}"
            f"  {count_before} -> {count_after}"
        )

    old, new = summarize(before_ranks), summarize(after_ranks)
    chunk_total = len(questions) * TOP_K
    print()
    print(f"{'Measure':<26} {'Before':>8} {'After':>8}")
    print(f"{'Expected-page Hit@5':<26} {old['hits']:>5}/{old['total']} {new['hits']:>5}/{new['total']}")
    print(f"{'Hit@5 rate':<26} {old['hit_rate']:>8.3f} {new['hit_rate']:>8.3f}")
    print(f"{'Expected-page MRR':<26} {old['mrr']:>8.3f} {new['mrr']:>8.3f}")
    print(f"{f'Correct-PDF chunks of {chunk_total}':<26} {right_pdf_before:>8} {right_pdf_after:>8}")
    print()
    print(f"Questions naming a paper: {sum(bool(find_named_papers(q['question'])) for q in questions)} of {len(questions)}")
    print(f"llm.py imported: {'llm' in sys.modules}")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\nAll pass criteria met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
