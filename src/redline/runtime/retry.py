"""Retry rate-limited model calls (HTTP 429), since Inspect doesn't for Groq.

Inspect 0.3.278's Groq provider returns a non-400 API error from `handle_bad_request` instead of
raising it, so `Model.generate` treats a 429 as final and never retries it (found in 1g.1, where
every cot cell on the 8,000 tokens-per-minute free tier failed at its first collision). This
wrapper retries only 429s: it waits as long as the provider's "try again in ..." asks, or backs
off exponentially, and gives up after `max_wait` seconds in total, so an exhausted daily quota
fails fast and the run resumes later. A failed call reports no usage, so budgets are unaffected.
"""

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable

# Private on purpose: Inspect doesn't export it, and it is the only error that keeps the
# provider's status code and message.
from inspect_ai.model._model import ModelGenerateError

logger = logging.getLogger(__name__)

RATE_LIMITED = 429
MAX_WAIT = 600.0  # seconds of waiting per call before giving up
BACKOFF_START = 5.0
BACKOFF_CAP = 60.0
_TRY_AGAIN = re.compile(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", re.IGNORECASE)

type Sleep = Callable[[float], Awaitable[None]]


def requested_wait(message: str | None) -> float | None:
    """Seconds from a provider message like "Please try again in 1m4.5s", if it says."""
    match = _TRY_AGAIN.search(message or "")
    if match is None or not any(match.groups()):
        return None
    hours, minutes, seconds = (float(part) if part else 0.0 for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


async def retry_rate_limits[T](
    call: Callable[[], Awaitable[T]], *, max_wait: float = MAX_WAIT, sleep: Sleep = asyncio.sleep
) -> T:
    """Await `call()`, retrying 429s until the waits would pass `max_wait` seconds in total."""
    waited = 0.0
    backoff = BACKOFF_START
    while True:
        try:
            return await call()
        except ModelGenerateError as error:
            if error.status_code != RATE_LIMITED:
                raise
            asked = requested_wait(error.provider_message)
            wait = (asked if asked is not None else backoff) + 0.5
            if waited + wait > max_wait:
                logger.warning("rate limited; giving up after %.0f s of waiting", waited)
                raise
            logger.info("rate limited; retrying in %.1f s", wait)
            await sleep(wait)
            waited += wait
            backoff = min(backoff * 2, BACKOFF_CAP)
