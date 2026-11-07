# RAG Document Assistant: Project Overview

Status as of 7 November 2026, on the local branch `add-tests-and-evaluation`.

**Summary:** the app still takes PDFs, searches them and has Gemini answer with page sources, but its search is now stronger. It uses token-based chunks, recognises paper names and reranks results with a local model. Retrieval improved in offline tests. Answer quality has not been re-measured on this setup, by choice, to avoid more API cost.

"Original" below means the GitHub version, commit `eeb0480`. The OneDrive folder also had some work that was never committed, which is pointed out where it matters.

## 1. Features in the original project

| Feature | What it does | File |
|---|---|---|
| **Read PDFs** | Pulls the text out of each page | `src/doc_loader.py` |
| **Chunking** | Cuts text into small overlapping pieces | `src/chunker.py` |
| **Embeddings** | Turns text into numbers that capture meaning | `src/embedder.py` |
| **Vector search** | Finds the pieces closest to a question | `src/vector_store.py`, `src/retriever.py` |
| **Ingestion** | Runs the steps above and saves the index | `src/ingest.py` |
| **Gemini answers** | Writes an answer from those pieces, and builds the source list | `src/llm.py` |
| **API** | Lets other programs ask questions through `/ask` | `src/main.py` |
| **Test script** | Asks one fixed question and prints the answer | `src/app.py` |
| **Docker** | Packages the API to run anywhere | `Dockerfile` |
| **Basic CI** | Checks Python syntax and builds Docker on GitHub | `.github/workflows/ci.yaml` |

**In OneDrive only:**

- `chat_ui.py`, the Streamlit page
- A 2-question evaluation file
- Streamlit in `requirements.txt`
- An active `COPY storage/` line in the `Dockerfile`

## 2. What's new, before and after

### Automated tests

- **Before:** the app had to be run and checked by eye. CI only checked syntax.
- **After:** 101 tests check the code automatically, without a Gemini key, in under 30 seconds.
- **Why:** a change that breaks something shows up straight away.

### Streamlit in the repo

This is the project owner's own work, not a new feature from this update.

- **Before:** it was only in OneDrive.
- **After:** the file is copied into the project, unchanged. `requirements-ui.txt` lists the packages it needs.

### Evaluation questions

- **Before:** 2 nearly identical questions on one page.
- **After:** 24 questions, 4 per PDF, each with an answer, a page and an exact quote.
- **Why:** this is the answer key the app gets graded against.

### Dataset checker (`evals/validate_dataset.py`)

- **Before:** nothing confirmed that the answers matched the PDFs.
- **After:** it fails if a quote isn't on its stated page, a number isn't in its quote, or a PDF doesn't have 4 questions.

### Ragas evaluation (`evals/run_ragas.py`)

- **Before:** there was no way to measure answer quality.
- **After:** the script scores each answer, saves results as it goes, and limits Gemini requests. All 24 questions have results, written up in `evals/RAG_Evaluation_Report.md`.

### Document-aware search (`src/retriever.py`)

- **Before:** all papers were searched together, so similar RAG papers got mixed up.
- **After:** if a question names a paper, such as "Corrective RAG" or "GPT-3", that paper's chunks come first. Otherwise the search works as before. Generic words such as "Transformer" don't count as a paper name.

### Token-based chunks (`src/chunker.py`)

- **Before:** chunks were 300 words. 399 of the 540 were longer than the 256 tokens the embedding model can read, so their endings were ignored in search.
- **After:** chunks are at most 224 tokens with a 32-token overlap, giving 1,011 chunks, all within the limit.

### Reranking (`src/reranker.py`)

- **Before:** the first search ranking was final.
- **After:** search finds 20 candidates, then a small local model (`cross-encoder/ms-marco-MiniLM-L6-v2`) reads each one with the question and keeps the best 5. It's free, runs on the CPU, and takes about 1 second per question.

## 3. Existing files changed

| File | Change | Why |
|---|---|---|
| `src/main.py` | Rejects blank questions, handles Gemini failures, uses the new index and reranking | Bug fixes and better search |
| `src/retriever.py` | Skips empty search results, and puts named papers first | Bug fix and better search |
| `src/chunker.py` | Measures chunks in tokens | Chunks now fit the model |
| `src/vector_store.py` | Adds the active index paths | Selects the new index |
| `src/ingest.py` | Saves rebuilds to `storage/token_224/` | A rebuild can't overwrite the original index |
| `src/app.py` | Uses the new index and reranking | Matches the API |
| `.github/workflows/ci.yaml` | Runs the tests | Catches breakage on GitHub |
| `README.md` | Adds setup, test, Docker and retrieval details | It was out of date |
| `.dockerignore` | Excludes `.venv/` and `evals/` | Keeps Docker builds small |
| `.gitignore` | Also covers index files in `storage/` subfolders | Keeps them out of Git |

**Unchanged from GitHub:** `Dockerfile`, `requirements.txt`, `src/doc_loader.py`, `src/embedder.py` and `src/llm.py`.

## 4. New files and folders

| Item | Purpose |
|---|---|
| `tests/` and `pytest.ini` | The automated tests |
| `requirements-dev.txt` | Packages for the tests |
| `requirements-ui.txt` | Packages for Streamlit |
| `requirements-eval.txt` | Packages for Ragas |
| `chat_ui.py` | The Streamlit page |
| `src/reranker.py` | Loads the reranker and picks the best chunks |
| `evals/eval_dataset.json` | The 24 questions |
| `evals/validate_dataset.py` | The dataset checker |
| `evals/run_ragas.py` | The Ragas script |
| `evals/results/ragas_results.jsonl` | Saved Ragas scores for the original search |
| `evals/RAG_Evaluation_Report.md` | Written report on those scores |
| `evals/check_retrieval.py` | Compares global and document-aware search, offline |
| `evals/compare_retrieval.py` | Compares chunking and reranking setups, offline |
| `PROJECT_OVERVIEW.md` | This document |

**On this computer only, and ignored by Git:** `.venv`, `evals/.venv`, `.env`, `data/` and the index files in `storage/`:

- `storage/token_224/`: the active 1,011-chunk index
- `storage/baseline_540/`: a backup of the original 540-chunk index
- `storage/faiss_index.bin` and `storage/chunks.pkl`: the original index, still used by `evals/run_ragas.py` and `evals/check_retrieval.py`

## 5. Bugs fixed

### In the app

| Bug | Before | After |
|---|---|---|
| Empty search slots | With few stored chunks, unrelated text could be added to the answer | Empty slots are skipped |
| Blank questions | Sent to search and Gemini anyway | Rejected with a 422 error |
| Gemini failures | Unhandled server error | A clear 503 message |
| Chunks too long | Most chunks were cut off when embedded | All chunks fit the model |
| Repeated chunk endings | A page could end with a chunk that only repeated the previous one | No longer happens |

### In the evaluation script (not the app)

- **Hidden retries:** one failed call could send up to 9 requests. Now it sends 1.
- **Connection error:** a closed event loop stopped an early run at question 3. It's fixed, and the full 24-question evaluation then finished.

## 6. How it works now

### Building the index, done once

```
PDFs → extract text → token-based chunks → embeddings → saved FAISS index
```

### Answering a question

```
You type a question in Streamlit
  → FastAPI receives it and checks it isn't blank
  → the question becomes numbers (on your computer)
  → FAISS finds the 20 closest chunks, with any named paper first (on your computer)
  → the reranker keeps the best 5 (on your computer)
  → Gemini writes an answer from those chunks (over the internet)
  → you see the answer and its PDF pages
```

### Testing (no AI services used)

`pytest` checks the code with fake data, and the dataset checker compares the 24 questions with the PDFs.

### Comparing retrieval (local models only)

`evals/compare_retrieval.py` runs the 24 questions through each setup and checks whether the expected page and the verified quote were found.

### Evaluating answers (uses Gemini)

```
Each of the 24 questions
  → search and a Gemini answer
  → Ragas grades it → score saved
```

### What each technology is for

| Technology | Simple meaning | Type |
|---|---|---|
| **PyMuPDF** | Reads PDF text | Original |
| **Sentence Transformers** | Turns text into numbers, and runs the reranker | Original |
| **FAISS** | A fast search through those numbers | Original |
| **Gemini** | The AI that writes answers | Original |
| **FastAPI** | The door other programs use to reach the app | Original |
| **Docker** | A sealed box that runs the app the same anywhere | Original |
| **GitHub Actions** | Runs checks automatically on GitHub | Original |
| **Streamlit** | The chat page you see | From OneDrive |
| **Pytest** | Runs automated checks on the code | New |
| **Ragas** | A grader for RAG answers | New |
| **Cross-encoder** | Rereads search results and reorders them | New |

### Ragas in simple terms

A normal test checks whether the code runs, while Ragas checks whether the answers are good. It uses an AI to grade three things:

- **Faithfulness:** did the answer stick to the retrieved text, or make things up?
- **Context Precision:** were the retrieved chunks useful, and ranked well?
- **Answer Correctness:** does it match the verified answer?

Scores run from 0 to 1, and higher is better. Gemini is grading its own model family here, so scores may be a little generous.

## 7. Results

### Answer quality: original search only

Ragas scored the original setup: global search on the 540-chunk index.

| Metric | Average |
|---|---|
| Answer Correctness | 0.702 |
| Faithfulness | 0.938 |
| Context Precision | 0.581 |

**These scores don't describe the current app.** It hasn't been through Ragas, so whether its answers are better is still unknown.

### Retrieval: offline, on the same 24 questions

All of these use document-aware search and keep 5 chunks.

| Setup | Expected page in top 5 | MRR | Full quote retrieved | Average context tokens |
|---|---|---|---|---|
| 540 chunks | 20/24 | 0.612 | 13/24 | 1,934 |
| 1,011 chunks | 18/24 | 0.560 | 11/24 | 1,013 |
| 540 chunks + reranking | 22/24 | 0.722 | 18/24 | 2,071 |
| **1,011 chunks + reranking (active)** | **20/24** | **0.729** | **19/24** | **1,058** |

- **Reranking made the main difference.** New chunks alone made retrieval slightly worse.
- **Why this setup was chosen:** it retrieved the most full quotes with about half the context. It gets all five of the weakest Ragas answers their full quotes.
- **The trade-off:** the 540-chunk index with reranking finds two more expected pages (Q5 and Q10), and Q10 loses its evidence on the active setup.
- **Still without a full quote:** Q5, Q8, Q9, Q10 and Q21.
- **How far to trust it:** this is a small sample, where one question moves a score by about 4%, and the paper names were chosen after seeing the questions.

## 8. Completed and pending

### Completed

- [x] 24 verified questions, approved and committed
- [x] Ragas evaluation of the original setup, for all 24 questions
- [x] Document-aware search, token-based chunks and reranking, compared offline
- [x] The new setup added to the app, with the original index backed up
- [x] The real FastAPI app run offline on all 24 questions, with Gemini replaced by a stand-in

### Pending

- [ ] Push and merge. **GitHub `main` is unchanged, so nobody can see this work yet**
- [ ] See GitHub CI pass for the first time
- [ ] Ask one real question through FastAPI and Streamlit on the new setup
- [ ] Build the Docker image with the new setup. The earlier Docker test came before these changes
- [ ] Test `requirements-eval.txt` on a clean install
- [ ] Check the Gemini limits and model, because Google's page says this preview model was shut down, even though it still worked in testing
- [ ] If not done yet, run `docker rm rag-test-local-run`, since that stopped container still holds the API key

### Optional, if the budget allows

- [ ] Run Ragas on the current setup, to measure answer quality directly

### Known weaknesses

- Answer quality on the current setup hasn't been measured.
- `evals/run_ragas.py` still tests the original index, not the one the app uses.
- The paper name list is fixed, so a new PDF needs its own entry to be recognised by name.
- A new Docker container downloads the models when it starts, so it needs internet access.
- Listed sources aren't always the pages Gemini actually used.
