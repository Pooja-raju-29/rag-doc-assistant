# RAG Evaluation Report

**All 24 evaluation questions were scored with Ragas. The app stays close to its retrieved text, with average Faithfulness of 0.938, but retrieval is its main weakness, with average Context Precision of 0.581.** Five answers scored below 0.5 on Answer Correctness.

| | |
|---|---|
| **Results source** | `evals/results/ragas_results.jsonl` |
| **Questions evaluated** | 24 of 24 |
| **PDFs covered** | 6 of 6 (4 questions each) |
| **Evaluation tool** | Ragas 0.4.3 |
| **Answer and grading model** | `gemini-3.1-flash-lite-preview` |
| **Report date** | 7 November 2026 |

## 1. Average scores

Scores run from 0 to 1, and higher is better.

| Metric | Question it answers | Average | Median |
|---|---|---|---|
| **Answer Correctness** | Does the answer match the verified answer? | **0.702** | 0.813 |
| **Faithfulness** | Did the answer stick to the retrieved text? | **0.938** | 1.000 |
| **Context Precision** | Did useful chunks come first in the search results? | **0.581** | 0.500 |

The expected page was among the listed sources for **18 of 24** questions.

## 2. How the system and evaluation work

**The app:**

```
Question → embed the question (all-MiniLM-L6-v2) → FAISS finds the top 5 chunks
         → Gemini answers from those chunks → answer and source pages
```

The index holds 6 research papers, 214 pages and 540 chunks.

**The evaluation:**

1. Each question in `evals/eval_dataset.json` goes through the real search and the real Gemini answer step.
2. Ragas grades the answer against the verified answer and the retrieved text.
3. The result is saved before the next question starts.

The 24 verified answers each have a source page and an exact quote checked against the PDFs by `evals/validate_dataset.py`.

## 3. Results for all 24 questions

"Expected page" means the verified answer's source page was among the sources listed for that question.

| ID | PDF | Question topic | Answer Correctness | Faithfulness | Context Precision | Expected page found |
|---|---|---|---|---|---|---|
| 1 | Attention | Encoder layers and sub-layers | 0.985 | 1.000 | 1.000 | Yes |
| 2 | Attention | Number of attention heads | 0.803 | 1.000 | 1.000 | Yes |
| 3 | Attention | Learning rate and warmup_steps | **0.337** | 1.000 | 1.000 | Yes |
| 4 | Attention | Label smoothing | 0.824 | 1.000 | 0.500 | Yes |
| 5 | Corrective RAG | The three actions | 0.846 | 1.000 | 0.333 | **No** |
| 6 | Corrective RAG | When web search is used | **0.160** | **0.500** | **0.000** | Yes |
| 7 | Corrective RAG | The Correct action | **0.074** | 1.000 | **0.000** | **No** |
| 8 | Corrective RAG | Datasets used | 0.951 | 1.000 | 0.200 | **No** |
| 9 | Few-Shot Learners | GPT-3 parameters | 0.976 | 1.000 | 0.500 | **No** |
| 10 | Few-Shot Learners | Few-shot setting | 0.733 | 1.000 | 0.700 | Yes |
| 11 | Few-Shot Learners | 3-digit subtraction accuracy | 0.834 | **0.000** | 1.000 | Yes |
| 12 | Few-Shot Learners | Malicious uses | 0.760 | 1.000 | 0.333 | Yes |
| 13 | RAG Survey | The three RAG paradigms | 0.976 | 1.000 | 0.250 | Yes |
| 14 | RAG Survey | Naive RAG process | 0.990 | 1.000 | 0.200 | Yes |
| 15 | RAG Survey | Why rewrite queries | 0.627 | 1.000 | 1.000 | Yes |
| 16 | RAG Survey | Context Relevance | 1.000 | 1.000 | 1.000 | Yes |
| 17 | RAG for NLP | RAG-Sequence vs RAG-Token | **0.197** | 1.000 | **0.000** | Yes |
| 18 | RAG for NLP | Retriever and seq2seq model | 0.584 | 1.000 | 0.333 | **No** |
| 19 | RAG for NLP | Thorough Decoding | 0.828 | 1.000 | 0.500 | Yes |
| 20 | RAG for NLP | Jeopardy question generation | 1.000 | 1.000 | 1.000 | Yes |
| 21 | Training Language Models | 1.3B model vs 175B GPT-3 | 0.534 | 1.000 | 1.000 | **No** |
| 22 | Training Language Models | TruthfulQA results | **0.375** | 1.000 | 1.000 | Yes |
| 23 | Training Language Models | Training prompt counts | 0.547 | 1.000 | 0.750 | Yes |
| 24 | Training Language Models | Fine-tuning algorithm | 0.912 | 1.000 | 0.333 | Yes |

**PDF short names**

| Short name | File |
|---|---|
| Attention | `attention_is_all_you_need.pdf` |
| Corrective RAG | `Corrective Retrieval Augmented Generation.pdf` |
| Few-Shot Learners | `Language Models are Few-Shot Learners.pdf` |
| RAG Survey | `Retrieval-Augmented Generation for Lar.pdf` |
| RAG for NLP | `Retrieval-Augmented Generation for.pdf` |
| Training Language Models | `Training language models to follow instructions.pdf` |

## 4. Results by PDF

Sorted by Answer Correctness. Each PDF has 4 questions, so these are small samples.

| PDF | Answer Correctness | Faithfulness | Context Precision | Expected page found |
|---|---|---|---|---|
| RAG Survey | 0.898 | 1.000 | 0.612 | 4 of 4 |
| Few-Shot Learners | 0.826 | 0.750 | 0.633 | 3 of 4 |
| Attention | 0.737 | 1.000 | 0.875 | 4 of 4 |
| RAG for NLP | 0.652 | 1.000 | 0.458 | 3 of 4 |
| Training Language Models | 0.592 | 1.000 | 0.771 | 3 of 4 |
| **Corrective RAG** | **0.508** | 0.875 | **0.133** | **1 of 4** |

## 5. Best-performing and worst-performing questions

The saved results hold the answers, final scores and source pages, but not the retrieved text or the grader's reasoning. So the reasons below are likely explanations, not confirmed causes.

### Best

| ID | Topic | Answer Correctness | Faithfulness | Context Precision |
|---|---|---|---|---|
| 16 | Context Relevance | 1.000 | 1.000 | 1.000 |
| 20 | Jeopardy question generation | 1.000 | 1.000 | 1.000 |
| 1 | Encoder layers and sub-layers | 0.985 | 1.000 | 1.000 |

These three were strong on all metrics, and the expected page was listed first each time.

### Worst

Ranked by Answer Correctness.

| ID | Score | What the saved answer shows |
|---|---|---|
| 7 | 0.074 | **Wrong.** It said retrieved items "in Self-RAG" are replaced with processed internal knowledge, which describes a different method. The expected page wasn't retrieved. |
| 6 | 0.160 | **Incomplete.** It described why web search helps but didn't name the Incorrect and Ambiguous actions. Most listed sources came from other papers. |
| 17 | 0.197 | **Off target.** The expected page was listed first, but the answer's definition didn't clearly state the same-document versus different-document distinction. |
| 3 | 0.337 | **Mostly wrong.** It said there was no learning rate schedule information, then described another paper's schedules. It did give `warmup_steps = 4000`. |
| 22 | 0.375 | **Missed the main point.** It focused on non-adversarial prompts and an "Instruction+QA" prompt, rather than the small improvements and the 1.3B PPO-ptx exception. |

### Scores that may understate the answer

| ID | Score | Why it may be low |
|---|---|---|
| 11 | Faithfulness 0.000 | See section 6 |
| 12 | Answer Correctness 0.760 | The answer nearly repeats the verified answer |
| 18 | Answer Correctness 0.584 | It names DPR and BART correctly and adds extra detail |
| 23 | Answer Correctness 0.547 | It gives all three correct counts and adds further numbers |

Answer Correctness partly compares how answers are phrased and split into statements, so a low score still needs reading by hand.

## 6. Investigation: Question 11's Faithfulness of 0.000

**Finding: the answer is correct, so this is probably a false alarm from the grader. The saved results can't confirm that.**

| Saved evidence | Value |
|---|---|
| Question | In the few-shot setting, what accuracy did GPT-3 achieve at 3-digit subtraction? |
| App's full answer | `94.2%` |
| Verified answer | GPT-3 achieved 94.2% accuracy at 3-digit subtraction in the few-shot setting. |
| Sources | Few-Shot Learners pages 22, 5 and 19 |
| Answer Correctness | 0.834 |
| Context Precision | 1.000 |

**How the score works.** In Ragas 0.4.3, Faithfulness has two steps:

1. An LLM turns the answer into separate statements.
2. An LLM checks each statement against the retrieved text. The score is the share that are supported.

If no statements are produced, the code returns "not a number" rather than 0. So a score of 0.000 means **statements were produced, and none were judged supported**.

**What the evidence suggests:**

- **The number is right.** It matches the verified answer, and the verified quote is on page 22.
- **Page 22 was listed first,** and Context Precision was 1.000.
- **The answer is unusually short.** It's the only saved answer that is just a number. The statement step has to rebuild a full claim from "94.2%", and a rebuilt claim may not match the retrieved text.

**What can't be confirmed.** The extracted statements, the grader's verdict and the retrieved text weren't saved. So the exact cause, and whether the retrieved part of page 22 held the figure, remain unknown.

**Effect on the average.** Average Faithfulness is 0.938 as recorded. It would be 0.979 if this one score were 1.000. The recorded score stays in the report, because no score was changed by hand.

## 7. Main retrieval weaknesses

1. **The three RAG papers get mixed up.** Corrective RAG had the lowest Context Precision (0.133) and found the expected page for only 1 of 4 questions. Its questions often listed RAG Survey pages first.
2. **Other papers often appear in the results.** 14 of the 24 questions listed at least one page from a different PDF (IDs 3, 5, 6, 7, 8, 9, 12, 15, 17, 18, 19, 20, 21 and 24).
3. **The expected page was missed for 6 questions:** IDs 5, 7, 8, 9, 18 and 21.
4. **Useful chunks often rank low.** 10 of the 24 questions scored 0.333 or below on Context Precision (IDs 5, 6, 7, 8, 12, 13, 14, 17, 18 and 24), and 3 scored 0.000 (IDs 6, 7 and 17).
5. **Finding the page doesn't guarantee a good answer.** Questions 3, 17 and 22 listed the expected page but still scored below 0.4 on Answer Correctness. This may come from how pages are chunked or from the answer step, and it needs checking separately.

**A correct answer can hide weak search.** Questions 8, 13 and 14 scored above 0.95 on Answer Correctness, with Context Precision of 0.250 or below. So answer scores alone don't show retrieval quality.

## 8. Recommended improvements

None of these have been tested yet, so each should be measured against these scores.

| Priority | Improvement | Problem it targets |
|---|---|---|
| 1 | Save retrieved text, sub-scores and grader reasoning | Low scores can't be explained directly (IDs 3, 11, 17) |
| 2 | Filter search by paper when a question names one | Papers getting mixed up (Corrective RAG) |
| 3 | Rerank the retrieved chunks | Useful chunks ranking low |
| 4 | Try hybrid search (keywords and embeddings) | Exact terms such as "Correct action" and "RAG-Token" |
| 5 | Test different chunk sizes and overlaps | Good page, poor answer (IDs 3, 17, 22) |
| 6 | Tell the answer step to answer in full sentences and keep to one paper | One-number replies and mixed sources (IDs 3, 7, 11) |
| 7 | List only the pages that support the answer | Sources include all retrieved pages |
| 8 | Add limited retries for temporary errors | Runs stopped twice on HTTP 503 errors |
| 9 | Try a different model family as the grader | Gemini grades its own family |

## 9. Limits of this evaluation

- **Small sample:** with 24 questions, one result moves an average by about 4%.
- **One run only:** each question was scored once, and LLM grading can vary.
- **Same model family:** Gemini wrote and graded the answers, so scores may be generous.
- **Single expected page:** a fact can appear on several pages, so "expected page not found" doesn't always mean the answer lacks support.
- **Different run conditions:** question 1 ran first, with no pause and with retries still on. The others ran with a 6-second pause and retries off.
- **Interrupted runs:** runs stopped on HTTP 429, HTTP 503 and a script connection bug before all questions finished. Unfinished questions were rerun, and finished results weren't changed.

## 10. Simple explanation for interviews

### In 30 seconds

> I built a RAG assistant over six AI research papers, then evaluated it with Ragas on 24 questions whose answers I verified against the PDFs. It scored 0.94 on Faithfulness, so it mostly sticks to its sources. Answer Correctness was 0.70 and Context Precision was 0.58, which pointed to retrieval as the main weakness. The biggest problem was mixing up similar papers. Corrective RAG questions often retrieved pages from a RAG survey. Next, I'd add saved diagnostics, document filtering and reranking, then rerun the same evaluation to measure the change.

### Likely questions

**What did you find most interesting?**

> High Faithfulness doesn't mean a good answer. On question 7, the app scored 1.0 on Faithfulness but 0.07 on Answer Correctness, because it described the wrong method. That's why I use several metrics, and also read the answers myself.

**How did you create the test set?**

> It has 24 questions, four per paper. Each has a verified answer, a source page and an exact quote, and a script checks that the quote is on that page. AI helped draft them, and I reviewed and approved them.

**Did you trust every score?**

> No. Question 11 scored 0 on Faithfulness although its answer, 94.2%, was correct. It was the only answer that was just a number, so the grader may have struggled with it. I kept the score, marked it as a likely false alarm, and noted that I can't confirm the cause without saving the grader's reasoning.

**What are the limits?**

> Twenty-four questions is a small sample, each was scored once, and Gemini graded its own model family. So I treat these as a baseline for comparing changes, not a precise measure of quality.

**What went wrong during the evaluation?**

> Testing found real issues. Hidden retries could turn one rate-limited call into nine requests, so I turned them off and counted every request. Gemini 503 errors also stopped runs, but because results were saved question by question, I could resume without repeating finished work.

## 11. Conclusion

**The pipeline works end to end and answers many questions well, but it isn't reliable yet on questions that require telling similar papers apart.** The strongest result is Faithfulness (0.938), and the clearest weakness is retrieval (Context Precision 0.581, and 0.133 for Corrective RAG).

These scores are a baseline. The next step is to save more diagnostics, improve retrieval, and rerun the same 24 questions to measure whether the changes help.
