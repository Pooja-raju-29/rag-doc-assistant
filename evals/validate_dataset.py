"""Validate the evaluation dataset against its rules and the source PDFs.

Run from the project root:

    python evals/validate_dataset.py

The structure checks need only this file and the dataset. The PDF checks also
need data/ and PyMuPDF, and are skipped with a clear message when data/ is
missing, because the PDFs are not in Git.
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

QUESTIONS_PER_PDF = 4
PDF_COUNT = 6
MIN_PAGES_PER_PDF = 3
REQUIRED_FIELDS = {"id", "question", "ground_truth", "source", "page", "evidence"}
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
REFERENCES_HEADING = re.compile(r"(^|\n)\s*(\d+\s+)?references\s*\n", re.IGNORECASE)


def normalize(text):
    """Make text comparable across PDF line breaks, ligatures and hyphenation."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s\-­‐‑]+", "", text)


def check_structure(items):
    """Return rule violations that can be found without the PDFs."""
    errors = []

    for number, item in enumerate(items, start=1):
        label = f"entry {item.get('id', number)}"
        missing = REQUIRED_FIELDS - item.keys()
        if missing:
            errors.append(f"{label}: missing fields {sorted(missing)}")
            continue

        for field in ("question", "ground_truth", "source", "evidence"):
            if not isinstance(item[field], str) or not item[field].strip():
                errors.append(f"{label}: {field} must be non-empty text")

        if not isinstance(item["page"], int) or item["page"] < 1:
            errors.append(f"{label}: page must be a positive whole number")

        if not errors or not errors[-1].startswith(label):
            evidence_numbers = set(NUMBER.findall(item["evidence"]))
            for value in NUMBER.findall(item["ground_truth"]):
                if value not in evidence_numbers:
                    errors.append(f"{label}: number {value} is not in the evidence")

    complete = [item for item in items if REQUIRED_FIELDS <= item.keys()]

    for value, count in Counter(item["id"] for item in complete).items():
        if count > 1:
            errors.append(f"id {value} is used {count} times")

    for value, count in Counter(normalize(item["question"]) for item in complete).items():
        if count > 1:
            errors.append(f"a question is repeated {count} times: {value[:50]}")

    per_source = Counter(item["source"] for item in complete)
    if len(per_source) != PDF_COUNT:
        errors.append(f"expected {PDF_COUNT} PDFs, found {len(per_source)}")

    for source, count in per_source.items():
        if count != QUESTIONS_PER_PDF:
            errors.append(f"{source}: expected {QUESTIONS_PER_PDF} questions, found {count}")
        pages = {item["page"] for item in complete if item["source"] == source}
        if len(pages) < MIN_PAGES_PER_PDF:
            errors.append(f"{source}: expected at least {MIN_PAGES_PER_PDF} pages, found {len(pages)}")

    return errors


def check_pdfs(items, data_dir):
    """Check every quote against its stated PDF page.

    Returns errors, plus notes about quotes that also appear on another page.
    """
    import pymupdf

    errors = []
    notes = []
    documents = {}
    # check_structure reports incomplete entries, so skip them here
    items = [item for item in items if REQUIRED_FIELDS <= item.keys()]

    for source in {item["source"] for item in items}:
        path = Path(data_dir) / source
        if not path.is_file():
            errors.append(f"{source}: PDF not found in {data_dir}")
            continue
        with pymupdf.open(path) as document:
            documents[source] = [page.get_text() for page in document]

    for item in items:
        pages = documents.get(item["source"])
        if pages is None:
            continue

        label = f"entry {item['id']}"
        page = item["page"]
        if page > len(pages):
            errors.append(f"{label}: page {page} is past the end of {item['source']}")
            continue

        references = next(
            (number for number, text in enumerate(pages, start=1) if REFERENCES_HEADING.search(text)),
            None,
        )
        if references is not None and page >= references:
            errors.append(f"{label}: page {page} is at or after the references (page {references})")

        evidence = normalize(item["evidence"])
        if evidence not in normalize(pages[page - 1]):
            errors.append(f"{label}: evidence not found on {item['source']} page {page}")

        others = [
            number
            for number, text in enumerate(pages, start=1)
            if number != page and evidence in normalize(text)
        ]
        if others:
            notes.append(f"{label}: evidence also appears on pages {others}")

    return errors, notes


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=root / "evals" / "eval_dataset.json")
    parser.add_argument("--data-dir", default=root / "data")
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as file:
        items = json.load(file)

    errors = check_structure(items)
    print(f"Structure checks: {len(items)} entries, {len(errors)} problems")

    if Path(args.data_dir).is_dir():
        pdf_errors, notes = check_pdfs(items, args.data_dir)
        errors += pdf_errors
        print(f"PDF checks: {len(pdf_errors)} problems")
        for note in notes:
            print(f"NOTE {note}")
    else:
        print(f"PDF checks: skipped, {args.data_dir} not found")

    for error in errors:
        print(f"ERROR {error}")

    print("RESULT: PASS" if not errors else "RESULT: FAIL")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
