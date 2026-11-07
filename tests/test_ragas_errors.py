import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

import asyncio  # noqa: E402

import httpx  # noqa: E402

from run_ragas import (  # noqa: E402
    METRICS,
    CallLimitReached,
    CountingTransport,
    Usage,
    describe_stop,
    evaluate_questions,
    rate_limit_info,
    redact,
)

FAKE_KEY = "placeholder-test-key-not-real"


class Result:
    value = 1.0


class RecordingMetric:
    """Record the event loop used, and optionally fail for one question."""

    def __init__(self, loops, usage=None, error=None, fail_question=None):
        self.loops = loops
        self.usage = usage
        self.error = error
        self.fail_question = fail_question

    async def ascore(self, user_input, **kwargs):
        self.loops.add(id(asyncio.get_running_loop()))
        if self.usage:
            self.usage.record("scoring request")
        if user_input == self.fail_question:
            raise self.error
        return Result()


def question(number):
    return {"id": number, "source": "s", "page": 1, "question": f"q{number}", "ground_truth": "a"}


def run_questions(numbers, metric, usage, done=()):
    saved = []
    asyncio.run(evaluate_questions(
        [question(number) for number in numbers],
        set(done),
        usage,
        {name: metric for name in METRICS},
        lambda item: ([{"text": "t"}], True),
        lambda text, chunks: "a",
        lambda chunks: [],
        saved.append,
        "test-model",
    ))
    return [record["id"] for record in saved]


def test_all_questions_share_one_event_loop():
    loops = set()

    assert run_questions([1, 2, 3], RecordingMetric(loops), Usage(50)) == [1, 2, 3]
    assert len(loops) == 1


def test_finished_questions_are_skipped_on_resume():
    assert run_questions([1, 2, 3], RecordingMetric(set()), Usage(50), done=[1, 3]) == [2]


def test_rate_limit_stops_the_run_and_keeps_finished_questions(capsys):
    error = RateLimitError(google_error("GenerateRequestsPerMinute-FreeTier"))
    metric = RecordingMetric(set(), error=error, fail_question="q2")

    assert run_questions([1, 2, 3], metric, Usage(50)) == [1]
    assert "Quota window: per-minute" in capsys.readouterr().out


def test_request_limit_stops_the_run(capsys):
    usage = Usage(5)

    saved = run_questions([1, 2, 3], RecordingMetric(set(), usage=usage), usage)

    assert saved == [1]
    assert usage.total == 5
    assert "reached the limit of 5" in capsys.readouterr().out


def send_through_counter(limit, attempts):
    """Send requests through the counter, returning how many reached the network."""
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(429)

    usage = Usage(limit)
    transport = CountingTransport(httpx.MockTransport(handler), usage)

    async def run():
        async with httpx.AsyncClient(transport=transport) as client:
            for _ in range(attempts):
                await client.post("https://example.test/", content=b"x")

    return sent, usage, run


def test_counter_counts_every_request_including_failed_ones():
    sent, usage, run = send_through_counter(limit=5, attempts=3)

    asyncio.run(run())

    assert len(sent) == 3
    assert usage.total == 3


def test_request_over_the_limit_is_never_sent():
    sent, usage, run = send_through_counter(limit=2, attempts=3)

    with pytest.raises(CallLimitReached):
        asyncio.run(run())

    assert len(sent) == 2
    assert usage.total == 2


class RateLimitError(Exception):
    status_code = 429

    def __init__(self, body, message="Error code: 429"):
        super().__init__(message)
        self.body = body


def google_error(*quota_ids, value="15", delay="42s"):
    return {
        "code": 429,
        "message": "You exceeded your current quota.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{"quotaId": quota, "quotaValue": value} for quota in quota_ids]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
        ],
    }


@pytest.mark.parametrize(
    "quota_ids, expected",
    [
        (["GenerateRequestsPerMinutePerProjectPerModel-FreeTier"], "per-minute"),
        (["GenerateRequestsPerDayPerProjectPerModel-FreeTier"], "daily"),
        (
            ["GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "GenerateRequestsPerDayPerProjectPerModel-FreeTier"],
            "per-minute and daily",
        ),
    ],
)
def test_quota_window_comes_from_google_quota_ids(quota_ids, expected):
    assert rate_limit_info(RateLimitError(google_error(*quota_ids)))["window"] == expected


def test_quota_ids_beat_other_wording_in_the_message():
    error = RateLimitError(
        google_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
        message="See limits per minute in the docs.",
    )

    assert rate_limit_info(error)["window"] == "daily"


def test_unknown_window_is_not_guessed():
    assert rate_limit_info(RateLimitError({"message": "Too many requests"}))["window"] == "unknown"


def test_limit_and_retry_delay_are_reported():
    info = rate_limit_info(RateLimitError(google_error("GenerateRequestsPerMinute-FreeTier")))

    assert info["limit"] == "15"
    assert info["retry_after"] == "42s"


def test_wrapped_rate_limit_is_found():
    try:
        try:
            raise RateLimitError(google_error("GenerateRequestsPerDay-FreeTier"))
        except RateLimitError as inner:
            raise RuntimeError("scoring failed") from inner
    except RuntimeError as outer:
        assert rate_limit_info(outer)["window"] == "daily"


def test_other_errors_are_not_called_rate_limits():
    assert rate_limit_info(ValueError("bad reply")) is None


def test_advice_for_a_daily_limit_says_a_pause_will_not_help():
    report = describe_stop(9, RateLimitError(google_error("GenerateRequestsPerDay-FreeTier")))

    assert "Quota window: daily" in report
    assert "will not help" in report


def test_api_key_never_appears_in_a_rate_limit_report():
    body = google_error("GenerateRequestsPerMinute-FreeTier")
    body["message"] = f"Quota exceeded for key={FAKE_KEY}"

    report = describe_stop(9, RateLimitError(body, message=f"Error using {FAKE_KEY}"), FAKE_KEY)

    assert FAKE_KEY not in report
    assert "[REDACTED]" in report


def test_api_key_never_appears_in_other_error_reports():
    assert FAKE_KEY not in describe_stop(9, ValueError(f"bad key {FAKE_KEY}"), FAKE_KEY)


@pytest.mark.parametrize(
    "text",
    [f"url?key={FAKE_KEY}&x=1", "key=custom-secret", "Authorization: Bearer custom-secret"],
)
def test_redact_removes_key_formats(text):
    result = redact(text)

    assert FAKE_KEY not in result
    assert "custom-secret" not in result
