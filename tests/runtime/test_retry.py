"""Rate-limited calls are retried with the provider's requested wait; nothing else is."""

import asyncio

import pytest
from inspect_ai.model._model import ModelGenerateError

from redline.runtime.retry import requested_wait, retry_rate_limits

TPM = (
    "Rate limit reached for model `openai/gpt-oss-20b` ... on tokens per minute (TPM): "
    "Limit 8000, Used 4272, Requested 4307. Please try again in 4.342499999s."
)


def limited(message: str | None = TPM, status: int = 429) -> ModelGenerateError:
    return ModelGenerateError("request failed", status_code=status, provider_message=message)


@pytest.mark.parametrize(
    ("message", "seconds"),
    [
        (TPM, 4.342499999),
        ("Please try again in 1m4.5s.", 64.5),
        ("try again in 2h3m", 7380.0),
        ("no hint here", None),
        (None, None),
    ],
)
def test_requested_wait(message: str | None, seconds: float | None) -> None:
    assert requested_wait(message) == (pytest.approx(seconds) if seconds else None)


def run(outcomes: list[object], max_wait: float = 600.0) -> tuple[object, list[float]]:
    """Call a fake model that raises or returns `outcomes` in order; record the sleeps."""
    queue = iter(outcomes)
    sleeps: list[float] = []

    async def call() -> object:
        outcome = next(queue)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    result = asyncio.run(retry_rate_limits(call, max_wait=max_wait, sleep=sleep))
    return result, sleeps


def test_a_429_waits_as_asked_then_succeeds() -> None:
    result, sleeps = run([limited(), limited("no hint"), "ok"])
    assert result == "ok"
    # the asked wait, then the backoff (5 s, doubled per attempt), each plus 0.5 s of margin
    assert sleeps == [pytest.approx(4.8425), 10.5]


def test_other_errors_are_not_retried() -> None:
    with pytest.raises(ModelGenerateError):
        run([limited(status=400), "never"])


def test_a_wait_past_the_cap_gives_up() -> None:
    # A spent daily quota asks for hours: give up at once, so the run can resume later.
    with pytest.raises(ModelGenerateError):
        run([limited("Please try again in 35m12s."), "never"], max_wait=600.0)
