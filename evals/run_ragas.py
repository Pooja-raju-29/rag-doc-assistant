"""Evaluate the RAG pipeline with Ragas.

Run from any folder, using the separate evaluation environment:

    evals/.venv/Scripts/python evals/run_ragas.py --dry-run
    evals/.venv/Scripts/python evals/run_ragas.py --ids 1,9 --confirm-api-calls

--dry-run uses the real FAISS index and local embeddings, but replaces Gemini
with a stand-in. It blocks outside network traffic and never reads .env. Its
scores are meaningless: it only tests the pipeline and counts the calls.

A real run makes Gemini API calls, so it needs --confirm-api-calls. Results are
saved after each question, and finished questions are skipped on the next run.
"""
import argparse
import asyncio
import json
import os
import re
import socket
import sys
import threading
import time
import types
import typing as t
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

# Ragas sends usage analytics unless this is set before it is imported
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evals" / "eval_dataset.json"
RESULTS = ROOT / "evals" / "results" / "ragas_results.jsonl"
GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
JUDGE_MODEL = "gemini-3.1-flash-lite-preview"
TOP_K = 5
MAX_CALLS = 400
METRICS = ("faithfulness", "context_precision", "answer_correctness")
CHARS_PER_TOKEN = 4
ASSUMED_OUTPUT_TOKENS = 200
INPUT_PRICE_PER_MILLION = 0.25
OUTPUT_PRICE_PER_MILLION = 1.50
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}
MAX_ERROR_CHARS = 500
FAKE_KEY = "dry-run-not-a-real-key"
PER_MINUTE = re.compile(r"per[\s_\-]*minute|\bRPM\b|\bTPM\b", re.IGNORECASE)
PER_DAY = re.compile(r"per[\s_\-]*day|\bRPD\b|\bTPD\b|daily", re.IGNORECASE)
QUOTA_ID = re.compile(r"""quotaId['"]?\s*[:=]\s*['"]?([\w\-.]+)""")
QUOTA_VALUE = re.compile(r"""quotaValue['"]?\s*[:=]\s*['"]?(\d+)""")
RETRY_DELAY = re.compile(r"""retryDelay['"]?\s*[:=]\s*['"]?(\d+(?:\.\d+)?s)""")


class CallLimitReached(Exception):
    """Stops a run before it can exceed the agreed number of Gemini calls."""


class Usage:
    """Counts Gemini calls by the part of the run that made them."""

    def __init__(self, limit):
        self.limit = limit
        self.calls = Counter()
        self.chars = Counter()
        self.current = "unknown"

    @property
    def total(self):
        return sum(self.calls.values())

    def check_limit(self):
        if self.total >= self.limit:
            raise CallLimitReached(f"reached the limit of {self.limit} calls")

    def record(self, prompt):
        self.check_limit()
        self.calls[self.current] += 1
        self.chars[self.current] += len(prompt)


def redact(text, api_key=None):
    """Remove API keys from text before it is shown."""
    text = str(text)
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    text = re.sub(r"AIza[0-9A-Za-z_\-]+", "[REDACTED]", text)
    text = re.sub(r"(?i)\b(key=)[^\s&'\"]+", r"\1[REDACTED]", text)
    return re.sub(r"(?i)\b(bearer\s+)[^\s'\"]+", r"\1[REDACTED]", text)


def error_chain(error):
    """Yield an error and the errors it wraps, as client libraries often wrap them."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        attempt = getattr(error, "last_attempt", None)
        wrapped = attempt.exception() if attempt is not None else None
        error = wrapped or error.__cause__ or error.__context__


def error_text(error):
    """Collect the text of an error, including its structured response details."""
    parts = [str(error)]
    for name in ("body", "details", "response_json"):
        value = getattr(error, name, None)
        if value:
            parts.append(json.dumps(value, default=str))
    return " ".join(parts)


def quota_window(text):
    """Say whether quota text describes a per-minute or a daily limit."""
    windows = []
    if PER_MINUTE.search(text):
        windows.append("per-minute")
    if PER_DAY.search(text):
        windows.append("daily")
    return " and ".join(windows) or "unknown"


def rate_limit_info(error):
    """Return details if this error is a Gemini rate limit, otherwise None."""
    for item in error_chain(error):
        status = getattr(item, "status_code", None) or getattr(item, "code", None)
        text = error_text(item)
        if status != 429 and type(item).__name__ != "RateLimitError" and "RESOURCE_EXHAUSTED" not in text:
            continue
        quota_ids = sorted(set(QUOTA_ID.findall(text)))
        delay = RETRY_DELAY.search(text)
        limit = QUOTA_VALUE.search(text)
        return {
            # Prefer Google's quota ids, as the message text can link to other limits
            "window": quota_window(" ".join(quota_ids) or text),
            "quota_ids": quota_ids,
            "limit": limit.group(1) if limit else None,
            "retry_after": delay.group(1) if delay else None,
            "message": text,
        }
    return None


def describe_stop(item_id, error, api_key=None):
    """Explain why a run stopped, without ever showing the API key."""
    info = rate_limit_info(error)
    if info is None:
        message = redact(error, api_key)[:MAX_ERROR_CHARS]
        return f"Stopped at entry {item_id}: {type(error).__name__}: {message}"

    advice = {
        "per-minute": "Wait about a minute, then retry with a longer --pause.",
        "daily": "A longer --pause will not help. Retry after the daily quota resets, or check billing.",
        "per-minute and daily": "Both limits were reported. Check AI Studio before retrying.",
        "unknown": "Google's reply does not say which limit this is. Check AI Studio before retrying.",
    }[info["window"]]
    lines = [
        f"Stopped at entry {item_id}: Gemini rate limit (HTTP 429).",
        f"  Quota window: {info['window']}",
        f"  Quota ids: {', '.join(info['quota_ids']) or 'not given'}",
        f"  Limit: {info['limit'] or 'not given'}",
        f"  Retry after: {info['retry_after'] or 'not given'}",
        f"  Next step: {advice}",
        f"  Google's message: {redact(info['message'], api_key)[:MAX_ERROR_CHARS]}",
    ]
    return "\n".join(redact(line, api_key) for line in lines)


def block_outside_network():
    """Make any non-local connection fail, and keep a list of the attempts."""
    blocked = []
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect

    def guarded_getaddrinfo(host, *args, **kwargs):
        name = host.decode() if isinstance(host, bytes) else host
        if name is not None and str(name) not in LOCAL_HOSTS:
            blocked.append(str(name))
            raise OSError(f"dry run blocked network lookup: {name}")
        return real_getaddrinfo(host, *args, **kwargs)

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if self.family in (socket.AF_INET, socket.AF_INET6) and host not in LOCAL_HOSTS:
            blocked.append(str(host))
            raise OSError(f"dry run blocked connection: {host}")
        return real_connect(self, address)

    socket.getaddrinfo = guarded_getaddrinfo
    socket.socket.connect = guarded_connect
    return blocked


def build_stub(model):
    """Build a valid placeholder for a Ragas response model."""
    from pydantic import BaseModel

    def value_for(annotation):
        origin = t.get_origin(annotation)
        args = t.get_args(annotation)
        if origin in (list, t.List):
            return [value_for(args[0])]
        if origin is dict:
            return {}
        if origin in (t.Union, types.UnionType):
            return value_for(next(arg for arg in args if arg is not type(None)))
        if origin is t.Literal:
            return args[0]
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return build_stub(annotation)
        if annotation is bool:
            return True
        if annotation is int:
            return 1
        if annotation is float:
            return 1.0
        return "dry-run placeholder"

    return model(**{name: value_for(field.annotation) for name, field in model.model_fields.items()})


def make_embeddings(model):
    """Reuse the project's local embedding model, so Gemini is not needed for it."""
    from ragas.embeddings.base import BaseRagasEmbedding

    class LocalEmbeddings(BaseRagasEmbedding):
        def embed_text(self, text, **kwargs):
            return model.encode([text])[0].tolist()

        async def aembed_text(self, text, **kwargs):
            return self.embed_text(text, **kwargs)

    return LocalEmbeddings()


def make_stub_llm(usage):
    from ragas.llms.base import InstructorBaseRagasLLM

    class StubLLM(InstructorBaseRagasLLM):
        def generate(self, prompt, response_model):
            usage.record(prompt)
            return build_stub(response_model)

        async def agenerate(self, prompt, response_model):
            return self.generate(prompt, response_model)

    return StubLLM()


class CountingTransport(httpx.AsyncBaseTransport):
    """Count each HTTP request, and refuse any request over the limit before sending it."""

    def __init__(self, inner, usage):
        self.inner = inner
        self.usage = usage

    async def handle_async_request(self, request):
        self.usage.record(request.content.decode("utf-8", errors="replace"))
        return await self.inner.handle_async_request(request)

    async def aclose(self):
        await self.inner.aclose()


def make_gemini_judge(usage, model, api_key, pause, transport=None, base_url=GEMINI_OPENAI_URL):
    """Create the Ragas LLM: one HTTP request per call, with every request counted.

    The OpenAI client and Instructor both retry by default, and together they sent
    up to 9 requests for one rate-limited call, so both are turned off here.
    """
    from openai import AsyncOpenAI, DefaultAsyncHttpxClient
    from ragas.llms import llm_factory
    from ragas.llms.base import InstructorBaseRagasLLM

    counted = CountingTransport(transport or httpx.AsyncHTTPTransport(), usage)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
        http_client=DefaultAsyncHttpxClient(transport=counted),
    )
    inner = llm_factory(model, provider="openai", client=client, max_retries=1)

    class CountedLLM(InstructorBaseRagasLLM):
        def generate(self, prompt, response_model):
            raise NotImplementedError("the collections metrics only use agenerate")

        async def agenerate(self, prompt, response_model):
            usage.check_limit()
            if pause:
                await asyncio.sleep(pause)
            return await inner.agenerate(prompt, response_model)

        async def aclose(self):
            await client.close()

    return CountedLLM()


def make_metrics(llm, embeddings):
    from ragas.metrics.collections import AnswerCorrectness, ContextPrecisionWithReference, Faithfulness

    return {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "answer_correctness": AnswerCorrectness(llm=llm, embeddings=embeddings),
    }


async def score_question(item, answer, contexts, metrics, usage):
    arguments = {
        "faithfulness": {"user_input": item["question"], "response": answer, "retrieved_contexts": contexts},
        "context_precision": {"user_input": item["question"], "reference": item["ground_truth"], "retrieved_contexts": contexts},
        "answer_correctness": {"user_input": item["question"], "response": answer, "reference": item["ground_truth"]},
    }
    scores = {}
    for name in METRICS:
        usage.current = name
        result = await metrics[name].ascore(**arguments[name])
        scores[name] = result.value
    return scores


def load_pipeline():
    """Load only the local parts of the app. src/llm.py needs a key, so it is separate."""
    sys.path.insert(0, str(ROOT / "src"))
    os.chdir(ROOT)
    from embedder import model
    from retriever import retrieve
    from vector_store import load_chunks, load_vector_store

    return model, retrieve, load_vector_store(), load_chunks()


def retrieve_for(item, model, retrieve, index, chunks):
    retrieved = retrieve(item["question"], model, index, chunks, top_k=TOP_K)
    page_found = any(c["source"] == item["source"] and c["page"] == item["page"] for c in retrieved)
    return retrieved, page_found


def completion_response(content):
    return httpx.Response(200, json={
        "id": "dry-run", "object": "chat.completion", "created": 0, "model": JUDGE_MODEL,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


def check_no_hidden_retries():
    """Use local stand-ins to confirm that one failing call sends one HTTP request.

    Returns one row per scenario: (name, requests sent, requests counted, passed).
    """
    from google import genai
    from ragas.metrics.collections.faithfulness.util import StatementGeneratorOutput as Output

    rate_limited = {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}

    def connection_failure(request):
        raise httpx.ConnectError("offline", request=request)

    scenarios = {
        "429 rate limit": lambda request: httpx.Response(429, json=rate_limited),
        "500 server error": lambda request: httpx.Response(500, json={"error": {"message": "server"}}),
        "connection failure": connection_failure,
        "unreadable reply": lambda request: completion_response("not json"),
    }
    rows = []

    for name, reply in scenarios.items():
        sent = []

        def handler(request, reply=reply, sent=sent):
            sent.append(request)
            return reply(request)

        usage = Usage(10)
        judge = make_gemini_judge(usage, JUDGE_MODEL, FAKE_KEY, 0, transport=httpx.MockTransport(handler))
        try:
            asyncio.run(judge.agenerate("Say ok.", Output))
            failed = False
        except Exception:
            failed = True
        rows.append((name, len(sent), usage.total, failed and len(sent) == 1 and usage.total == 1))

    # The limit must stop the request from being sent at all
    sent = []

    def accept(request):
        sent.append(request)
        return completion_response(json.dumps({"statements": ["ok"]}))

    usage = Usage(2)
    judge = make_gemini_judge(usage, JUDGE_MODEL, FAKE_KEY, 0, transport=httpx.MockTransport(accept))
    outcomes = []
    for _ in range(3):
        try:
            asyncio.run(judge.agenerate("Say ok.", Output))
            outcomes.append("sent")
        except CallLimitReached:
            outcomes.append("stopped")
    rows.append(("limit of 2, three calls", len(sent), usage.total, outcomes == ["sent", "sent", "stopped"] and len(sent) == 2))

    # The app's own client, with the same default settings as src/llm.py
    sent = []

    def google_reply(request):
        sent.append(request)
        return httpx.Response(429, json=rated_google_error())

    client = genai.Client(api_key=FAKE_KEY, http_options=genai.types.HttpOptions(client_args={"transport": httpx.MockTransport(google_reply)}))
    try:
        client.models.generate_content(model=JUDGE_MODEL, contents="Say ok.")
        failed = False
    except Exception as error:
        failed = rate_limit_info(error) is not None
    rows.append(("app answer client, 429", len(sent), 1, failed and len(sent) == 1))
    return rows


def rated_google_error():
    return {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}


def check_real_connection_offline():
    """Send one fake request through the real Ragas and OpenAI client code path.

    A local stand-in answers it, so this checks the wiring without using Gemini.
    """
    from ragas.metrics.collections.faithfulness.util import StatementGeneratorOutput

    requests = []

    def handler(request):
        requests.append(request)
        return completion_response(json.dumps({"statements": ["ok"]}))

    judge = make_gemini_judge(Usage(1), JUDGE_MODEL, FAKE_KEY, 0, transport=httpx.MockTransport(handler))
    result = asyncio.run(judge.agenerate("Say ok.", StatementGeneratorOutput))
    request = requests[0]
    return {
        "requests": len(requests),
        "path": request.url.path,
        "model_sent": json.loads(request.content)["model"],
        "parsed": result.statements == ["ok"],
    }


async def evaluate_questions(items, done, usage, metrics, find_chunks, answer_question, get_sources, save, judge_model, api_key=None):
    """Score each question in one event loop, saving results as they finish.

    The HTTP client keeps connections open, and they belong to the event loop that
    made them, so a new loop per question would break the next question's request.
    """
    for item in items:
        if item["id"] in done:
            print(f"entry {item['id']}: already done, skipped")
            continue
        retrieved, page_found = find_chunks(item)
        contexts = [chunk["text"] for chunk in retrieved]
        started = time.perf_counter()
        try:
            usage.current = "app_answer"
            usage.record("\n\n".join(contexts) + item["question"])
            answer = answer_question(item["question"], retrieved)
            scores = await score_question(item, answer, contexts, metrics, usage)
        except Exception as error:
            if any(isinstance(wrapped, CallLimitReached) for wrapped in error_chain(error)):
                print(f"Stopped: reached the limit of {usage.limit} calls. Finished questions are saved.")
                return
            print(describe_stop(item["id"], error, api_key))
            print("Finished questions are saved.")
            return

        save({
            "id": item["id"],
            "source": item["source"],
            "page": item["page"],
            "page_in_top_k": page_found,
            "answer": answer,
            "sources": get_sources(retrieved),
            "scores": scores,
            "seconds": round(time.perf_counter() - started, 1),
            "judge_model": judge_model,
        })
        print(f"entry {item['id']}: {scores}")


def check_connections_across_questions():
    """Run several questions through real HTTP connections to a local stand-in server.

    MockTransport keeps no connections, so it cannot catch connection reuse problems.
    Returns the questions saved, the requests received and the connections opened.
    """
    from ragas.metrics.collections.faithfulness.util import StatementGeneratorOutput

    received = []
    connections = set()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            received.append(self.path)
            connections.add(self.client_address)
            reply = completion_response(json.dumps({"statements": ["ok"]})).content
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/v1/"
    usage = Usage(50)
    judge = make_gemini_judge(usage, JUDGE_MODEL, FAKE_KEY, 0, base_url=url)

    class OneRequestMetric:
        async def ascore(self, **kwargs):
            await judge.agenerate("Say ok.", StatementGeneratorOutput)
            return types.SimpleNamespace(value=1.0)

    metrics = {name: OneRequestMetric() for name in METRICS}
    questions = [
        {"id": number, "source": "s", "page": 1, "question": "q", "ground_truth": "a"}
        for number in (1, 2, 3)
    ]
    saved = []

    async def run():
        try:
            await evaluate_questions(
                questions, set(), usage, metrics, lambda item: ([{"text": "t"}], True),
                lambda question, chunks: "a", lambda chunks: [], saved.append, JUDGE_MODEL, FAKE_KEY,
            )
        finally:
            await judge.aclose()

    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
    return len(saved), len(received), len(connections)


def dry_run(items):
    blocked = block_outside_network()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model, retrieve, index, chunks = load_pipeline()
    usage = Usage(limit=10_000)
    metrics = make_metrics(make_stub_llm(usage), make_embeddings(model))

    failures = []
    pages_found = 0
    for item in items:
        retrieved, page_found = retrieve_for(item, model, retrieve, index, chunks)
        pages_found += page_found
        contexts = [chunk["text"] for chunk in retrieved]
        usage.current = "app_answer"
        usage.record("\n\n".join(contexts) + item["question"])
        try:
            scores = asyncio.run(score_question(item, "Dry-run placeholder answer.", contexts, metrics, usage))
            if set(scores) != set(METRICS):
                failures.append(f"entry {item['id']}: missing scores")
        except Exception as error:
            failures.append(f"entry {item['id']}: {type(error).__name__}: {error}")

    wiring = check_real_connection_offline()
    app_client_loaded = "llm" in sys.modules
    retry_rows = check_no_hidden_retries()
    saved, received, connections = check_connections_across_questions()
    connections_ok = saved == 3 and received == 9
    total = usage.total
    input_tokens = sum(usage.chars.values()) / CHARS_PER_TOKEN
    output_tokens = total * ASSUMED_OUTPUT_TOKENS
    cost = (input_tokens * INPUT_PRICE_PER_MILLION + output_tokens * OUTPUT_PRICE_PER_MILLION) / 1_000_000

    print(f"Questions processed: {len(items)}, failures: {len(failures)}")
    for failure in failures:
        print(f"  FAILED {failure}")
    print(f"Correct page in top {TOP_K} (real local search): {pages_found}/{len(items)}")
    print(f"Connection check through the real client, against a local stand-in: {wiring}")
    print(f"Outside network attempts blocked: {len(blocked)} {sorted(set(blocked))}")
    print(f"src/llm.py loaded: {app_client_loaded}")
    print("\nHidden retry checks (scenario | requests sent | requests counted | passed):")
    for name, sent, counted, passed in retry_rows:
        print(f"  {name} | {sent} | {counted} | {'PASS' if passed else 'FAIL'}")
    print(
        "\nReal connections across questions, against a local server: "
        f"{saved}/3 questions saved, {received}/9 requests received, "
        f"{connections} connection(s) opened: {'PASS' if connections_ok else 'FAIL'}"
    )
    print("\nExpected Gemini requests, with no automatic retries:")
    for name in ("app_answer", *METRICS):
        print(f"  {name}: {usage.calls[name]} ({usage.calls[name] / len(items):g} per question)")
    print(f"  TOTAL: {total}")
    print(f"\nEstimated tokens: about {input_tokens:,.0f} input, {output_tokens:,.0f} output")
    print(f"Estimated cost at paid 3.1 Flash-Lite rates: about ${cost:.2f}")
    print("Stand-in scores were discarded, because they mean nothing.")
    passed = not failures and not blocked and connections_ok and all(row[3] for row in retry_rows)
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


def real_run(items, args):
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set, so nothing was run.")
        return 1

    model, retrieve, index, chunks = load_pipeline()
    from llm import generate_answer, get_sources

    usage = Usage(limit=args.max_calls)
    judge = make_gemini_judge(usage, args.judge_model, api_key, args.pause)
    metrics = make_metrics(judge, make_embeddings(model))

    done = set()
    if RESULTS.exists():
        with open(RESULTS, encoding="utf-8") as file:
            done = {json.loads(line)["id"] for line in file if line.strip()}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    def save(record):
        with open(RESULTS, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def run():
        try:
            await evaluate_questions(
                items, done, usage, metrics,
                lambda item: retrieve_for(item, model, retrieve, index, chunks),
                generate_answer, get_sources, save, args.judge_model, api_key,
            )
        finally:
            await judge.aclose()

    asyncio.run(run())
    print(f"Gemini calls this run: {usage.total} {dict(usage.calls)}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="make no API calls")
    parser.add_argument("--confirm-api-calls", action="store_true", help="allow real Gemini calls")
    parser.add_argument("--ids", help="comma-separated question ids, such as 1,9")
    parser.add_argument("--max-calls", type=int, default=MAX_CALLS)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--pause", type=float, default=0, help="seconds to wait before each scoring call")
    args = parser.parse_args()

    with open(DATASET, encoding="utf-8") as file:
        items = json.load(file)
    if args.ids:
        wanted = {int(value) for value in args.ids.split(",")}
        items = [item for item in items if item["id"] in wanted]

    if args.dry_run:
        return dry_run(items)
    if not args.confirm_api_calls:
        print("This would make Gemini API calls. Use --dry-run, or add --confirm-api-calls.")
        return 2
    return real_run(items, args)


if __name__ == "__main__":
    sys.exit(main())
