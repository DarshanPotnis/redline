"""Find the last top-level JSON value of a given kind in free text: fences and prose are fine."""

import json
from typing import Any, Literal

_PLAIN = json.JSONDecoder()


def last_json(text: str, opener: Literal["{", "["], decoder: json.JSONDecoder = _PLAIN) -> Any:
    """The last JSON value starting with `opener` that decodes, or None.

    Every `opener` is tried, and the scan skips past each decoded value, so values nested inside
    an earlier one never count on their own.
    """
    found: Any = None
    start = text.find(opener)
    while start != -1:
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find(opener, start + 1)
            continue
        found = value
        start = text.find(opener, end)
    return found
