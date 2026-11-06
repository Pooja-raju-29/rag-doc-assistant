import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals"))

from validate_dataset import check_pdfs, check_structure, normalize  # noqa: E402

DATA_DIR = ROOT / "data"


@pytest.fixture
def items():
    with open(ROOT / "evals" / "eval_dataset.json", encoding="utf-8") as file:
        return json.load(file)


def test_dataset_follows_all_structure_rules(items):
    assert check_structure(items) == []


def test_dataset_has_24_questions(items):
    assert len(items) == 24


def test_missing_field_is_reported(items):
    del items[0]["evidence"]

    assert any("missing fields" in error for error in check_structure(items))


def test_duplicate_id_is_reported(items):
    items[1]["id"] = items[0]["id"]

    assert any("is used 2 times" in error for error in check_structure(items))


def test_wrong_question_count_is_reported(items):
    assert any("expected 4 questions" in error for error in check_structure(items[:-1]))


def test_number_missing_from_evidence_is_reported(items):
    items[1]["ground_truth"] = "The Transformer uses 9 heads."

    assert any("number 9" in error for error in check_structure(items))


def test_too_few_pages_for_a_pdf_is_reported(items):
    changed = copy.deepcopy(items)
    for item in changed:
        if item["source"] == changed[0]["source"]:
            item["page"] = 3

    assert any("at least 3 pages" in error for error in check_structure(changed))


def test_normalize_handles_pdf_line_breaks_and_ligatures():
    assert normalize("position- wise\nfine") == normalize("position-wise ﬁne")


def test_normalize_keeps_decimal_points():
    assert normalize("80.2%") != normalize("802%")


@pytest.mark.skipif(not DATA_DIR.is_dir(), reason="PDFs are not in Git")
def test_every_quote_is_on_its_stated_pdf_page(items):
    pytest.importorskip("pymupdf")

    errors, _ = check_pdfs(items, DATA_DIR)

    assert errors == []


@pytest.mark.skipif(not DATA_DIR.is_dir(), reason="PDFs are not in Git")
def test_wrong_page_is_detected(items):
    pytest.importorskip("pymupdf")
    items[0]["page"] = 2

    errors, _ = check_pdfs(items, DATA_DIR)

    assert any("evidence not found" in error for error in errors)
