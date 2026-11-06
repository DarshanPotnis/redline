"""Model profiles: everything that shapes a model's generations, pinned and logged.

Every real-model run goes through a registered profile. It records the budget (`max_tokens`:
TOTAL_BUDGET for every profile; BudgetedModel sets each call's cap within it), the context
window, thinking, the sampling sent with each request, and how many requests may run at once.
A local model's profile also records its Modelfile and the file's SHA-256. There are no silent
defaults: an unregistered model is an error.
"""

import functools
import hashlib
from pathlib import Path
from typing import Literal

from inspect_ai.model import GenerateConfig

# Private on purpose: the public get_model() falls back to INSPECT_EVAL_MODEL when no model is
# being evaluated, which would resolve a profile for a model that may never run.
from inspect_ai.model._model import active_model
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from redline.budget import TOTAL_BUDGET

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELFILES = REPO_ROOT / "models" / "ollama"
LOCAL_MODELS = ("qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b")


class ModelProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str  # the Inspect model ID
    provider: Literal["ollama", "groq", "mockllm"]
    context_window: int | None  # in tokens; None for the mock model, which has none
    max_tokens: int = TOTAL_BUDGET
    thinking: Literal["off", "on", "n/a"]
    max_connections: int | None = None  # concurrent requests; None: Inspect's default
    generate: dict[str, JsonValue] = Field(default_factory=dict)  # GenerateConfig overrides
    modelfile_path: str | None = None  # repo-relative; local models only
    modelfile_sha256: str | None = None

    def generate_config(self) -> GenerateConfig:
        return GenerateConfig.model_validate(
            {
                "max_tokens": self.max_tokens,
                "max_connections": self.max_connections,
                **self.generate,
            }
        )


@functools.cache
def profiles() -> dict[str, ModelProfile]:
    registered = [
        *(_local(name) for name in LOCAL_MODELS),
        ModelProfile(
            model_id="mockllm/model", provider="mockllm", context_window=None, thinking="n/a"
        ),
        # Context windows: console.groq.com/docs/models, read 2026-10-13.
        # Connections: 1. The free tier allows 8,000 tokens per minute (both models' headers,
        # 2026-11-05), and Groq reserves each request's max_tokens against it, so a 3,584-token
        # reasoning call takes about half a minute's quota: a second connection only collides.
        # 429s are retried by runtime/retry.py (Inspect doesn't, for Groq).
        # The non-reasoning Groq model since 1g.1; llama-3.3-70b-versatile is gone from the free
        # tier (404 model_not_found, 2026-11-05). Qwen 3.8 27B, console.groq.com/docs/model/qwen/
        # qwen3.8-27b, read 2026-11-05: 131,072-token context.
        # Thinking: Groq serves it non-thinking by default (checked 2026-11-05: 2 completion
        # tokens, no reasoning_tokens, with or without reasoning_effort "none"). Inspect's Groq
        # provider can't send reasoning_effort "none" (it clamps it away) or extra_body, so the
        # profile relies on that default; every call's logged reasoning_tokens shows it held.
        # Sampling: the model card's non-thinking settings. top_k=20 can't be sent (Groq rejects
        # top_k) and min_p=0 is the default.
        ModelProfile(
            model_id="groq/qwen/qwen3.8-27b",
            provider="groq",
            context_window=131_072,
            thinking="off",
            max_connections=1,
            generate={"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5, "seed": 0},
        ),
        # Sampling: "We recommend sampling with temperature=1.0 and top_p=1.0" (openai/gpt-oss
        # README). Its reasoning tokens are part of the completion, so they count toward the budget.
        ModelProfile(
            model_id="groq/openai/gpt-oss-20b",
            provider="groq",
            context_window=131_072,
            thinking="on",
            max_connections=1,
            generate={"temperature": 1.0, "top_p": 1.0, "seed": 0, "reasoning_effort": "medium"},
        ),
    ]
    return {profile.model_id: profile for profile in registered}


def profile_for(model_id: str) -> ModelProfile:
    known = profiles()
    if model_id not in known:
        raise ValueError(f"no model profile for {model_id!r}; known profiles: {', '.join(known)}")
    return known[model_id]


def active_profile() -> ModelProfile:
    """The profile of the model Inspect is evaluating right now."""
    model = active_model()
    if model is None:
        raise RuntimeError(
            "no model is being evaluated, so there is no profile to use: run the task through "
            "`inspect eval --model <id>` or `inspect_ai.eval(<task function>, model=...)`"
        )
    return profile_for(str(model))


def parse_modelfile(text: str) -> dict[str, str]:
    """The FROM line and the single-valued PARAMETER lines of a Modelfile.

    E.g. {"FROM": "qwen3.5:2b", "num_ctx": "8192", "temperature": "1.0"}.
    """
    entries: dict[str, str] = {}
    for line in text.splitlines():
        match line.split():
            case ["FROM", base]:
                entries["FROM"] = base
            case ["PARAMETER", name, value]:
                entries[name] = value
    return entries


def _local(name: str) -> ModelProfile:
    path = MODELFILES / f"{name}.Modelfile"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; local profiles read the repo's Modelfiles")
    content = path.read_bytes()
    parameters = parse_modelfile(content.decode())
    return ModelProfile(
        model_id=f"ollama/redline-{name}",
        provider="ollama",
        context_window=int(parameters["num_ctx"]),
        thinking="off",
        # One request at a time: Ollama otherwise serves several in parallel, and four 8k
        # contexts of the 4B model would swap on the 8 GB dev machine.
        max_connections=1,
        generate={
            # Ollama's OpenAI-compatible API sets temperature and top_p to 1.0 whenever a request
            # leaves them out (openai/openai.go, v0.33.3), overriding the Modelfile, so these are
            # read from it and sent. Seed, top_k, min_p and the penalties are only set when a
            # request sends them, so the Modelfile governs those.
            "temperature": float(parameters["temperature"]),
            "top_p": float(parameters["top_p"]),
            # Ollama maps reasoning_effort "none" to think=false. Inspect's own reasoning_effort
            # sends nothing for "none" with Ollama, so it goes into the request body directly.
            "extra_body": {"reasoning_effort": "none"},
        },
        modelfile_path=path.relative_to(REPO_ROOT).as_posix(),
        modelfile_sha256=hashlib.sha256(content).hexdigest(),
    )
