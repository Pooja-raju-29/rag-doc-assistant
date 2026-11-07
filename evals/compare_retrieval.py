"""Compare the retrieval setups on the 24 evaluation questions.

Runs on this computer only: no Gemini calls, and any outside connection is blocked.
It prints its results and writes no files. It needs the original index backup in
storage/baseline_540/ and the active index in storage/token_224/.

    .venv/Scripts/python.exe evals/compare_retrieval.py
"""
import json
import os
import socket
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = 20
TOP_K = 5
MIN_MATCH = 20  # normalized characters; shorter matches are treated as chance


def block_outside_network():
    def blocked(*args, **kwargs):
        raise OSError("compare_retrieval.py does not use the network")

    socket.getaddrinfo = blocked
    socket.socket.connect = blocked


def evidence_share(evidence, texts):
    """Share of the evidence found in the texts, allowing it to span chunks."""
    found = [False] * len(evidence)
    for text in texts:
        for block in SequenceMatcher(None, evidence, text, autojunk=False).get_matching_blocks():
            if block.size >= MIN_MATCH:
                found[block.a:block.a + block.size] = [True] * block.size
    return sum(found) / len(evidence)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    block_outside_network()

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "evals"))

    from check_retrieval import expected_rank, summarize
    from embedder import model
    from reranker import rerank
    from retriever import retrieve
    from validate_dataset import normalize
    from vector_store import load_chunks, load_vector_store

    indexes = {
        "old": (load_vector_store("storage/baseline_540/faiss_index.bin"), load_chunks("storage/baseline_540/chunks.pkl")),
        "new": (load_vector_store("storage/token_224/faiss_index.bin"), load_chunks("storage/token_224/chunks.pkl")),
    }
    questions = json.loads((ROOT / "evals" / "eval_dataset.json").read_text(encoding="utf-8"))
    names = ["old", "new", "old+rerank", "new+rerank"]
    results = {name: {} for name in names}

    def measure(found, item):
        evidence = normalize(item["evidence"])
        return {
            "rank": expected_rank(found, item["source"], item["page"]),
            "share": evidence_share(evidence, [normalize(chunk["text"]) for chunk in found]),
            "tokens": sum(
                len(model.tokenizer(chunk["text"], add_special_tokens=False, verbose=False)["input_ids"])
                for chunk in found
            ),
        }

    for item in questions:
        for label, (index, chunks) in indexes.items():
            candidates = retrieve(item["question"], model, index, chunks, top_k=CANDIDATES)
            results[label][item["id"]] = measure(candidates[:TOP_K], item)
            reranked = rerank(item["question"], candidates, top_k=TOP_K)
            results[f"{label}+rerank"][item["id"]] = measure(reranked, item)

    rows = {
        "Expected-page Hit@5": lambda r: f"{summarize([x['rank'] for x in r.values()])['hits']}/{len(r)}",
        "Expected-page MRR": lambda r: f"{summarize([x['rank'] for x in r.values()])['mrr']:.3f}",
        "Full evidence retrieved": lambda r: f"{sum(x['share'] >= 0.999 for x in r.values())}/{len(r)}",
        "Average context tokens": lambda r: f"{sum(x['tokens'] for x in r.values()) / len(r):.0f}",
    }
    print(f"{'Measure':<26}" + "".join(f"{name:>12}" for name in names))
    for label, cell in rows.items():
        print(f"{label:<26}" + "".join(f"{cell(results[name]):>12}" for name in names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
