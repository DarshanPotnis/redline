"""Keep at most one Ollama model in memory: the 8 GB dev machine can't hold two.

Ollama keeps a model loaded for minutes after its last request, so switching models would
briefly hold both. The runner clears every loaded model before a local model starts and after
it finishes.
"""

import json
import logging
import os
import time
import urllib.request
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434/v1"  # Inspect's ollama/ provider default


class ModelMemory(Protocol):
    def unload_all(self) -> list[str]:
        """Unload every loaded model and return their names."""
        ...


def api_root(base_url: str | None = None) -> str:
    """Ollama's native API root, from the OpenAI-compatible base URL Inspect uses."""
    base = base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL
    return base.rstrip("/").removesuffix("/v1")


class Ollama:
    def __init__(self, root: str | None = None, timeout: float = 60.0) -> None:
        self.root = root or api_root()
        self.timeout = timeout  # seconds to wait for unloaded models to leave memory

    def loaded(self) -> list[str]:
        response = self._request("GET", "/api/ps")
        return sorted(model["name"] for model in response.get("models", []))

    def unload_all(self) -> list[str]:
        names = self.loaded()
        for name in names:
            # An empty request with keep_alive 0 unloads the model (Ollama API docs).
            self._request("POST", "/api/generate", {"model": name, "keep_alive": 0})
        deadline = time.monotonic() + self.timeout
        while self.loaded():
            if time.monotonic() > deadline:
                raise RuntimeError(f"Ollama still holds {self.loaded()} after unloading")
            time.sleep(0.5)
        if names:
            logger.info("unloaded from Ollama: %s", ", ".join(names))
        return names

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.root + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read() or b"{}")
