"""Model profiles: registered, faithful to their Modelfiles, and exactly what goes on the wire."""

import hashlib
import re

import pytest
from inspect_ai.model import get_model

from redline.runtime.profiles import (
    LOCAL_MODELS,
    MAX_TOKENS,
    MODELFILES,
    REPO_ROOT,
    active_profile,
    parse_modelfile,
    profile_for,
    profiles,
)

LOCAL_IDS = [f"ollama/redline-{name}" for name in LOCAL_MODELS]
# Every sampling parameter a Modelfile must pin, so no Ollama default applies unseen.
PINNED = {
    "num_ctx",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repeat_penalty",
    "seed",
}


def test_the_registry() -> None:
    assert list(profiles()) == [
        *LOCAL_IDS,
        "mockllm/model",
        "groq/llama-3.3-70b-versatile",
        "groq/openai/gpt-oss-20b",
    ]


def test_every_profile_has_the_same_budget() -> None:
    assert MAX_TOKENS == 4096
    for profile in profiles().values():
        assert profile.max_tokens == profile.generate_config().max_tokens == MAX_TOKENS


def test_every_modelfile_has_a_profile_and_back() -> None:
    assert sorted(
        path.name.removesuffix(".Modelfile") for path in MODELFILES.glob("*.Modelfile")
    ) == sorted(LOCAL_MODELS)


@pytest.mark.parametrize("model_id", LOCAL_IDS)
def test_local_profiles_follow_their_modelfile(model_id: str) -> None:
    profile = profile_for(model_id)
    assert profile.modelfile_path is not None
    path = REPO_ROOT / profile.modelfile_path
    parameters = parse_modelfile(path.read_text())
    assert set(parameters) >= PINNED
    assert parameters["seed"] == "0"
    assert parameters["FROM"] == "qwen3.5:" + model_id.removeprefix("ollama/redline-qwen3.5-")
    assert profile.context_window == int(parameters["num_ctx"]) == 8192
    config = profile.generate_config()
    assert config.temperature == float(parameters["temperature"])
    assert config.top_p == float(parameters["top_p"])
    assert config.extra_body == {"reasoning_effort": "none"}
    assert profile.thinking == "off"
    assert profile.modelfile_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("model_id", LOCAL_IDS)
def test_the_request_turns_thinking_off_and_pins_sampling(model_id: str) -> None:
    # What Inspect's ollama provider will send, without calling a model.
    profile = profile_for(model_id)
    params = get_model(model_id).api.completion_params(profile.generate_config(), tools=False)
    assert params["extra_body"] == {"reasoning_effort": "none"}
    assert (params["temperature"], params["top_p"]) == (1.0, 1.0)
    assert params.get("max_tokens", params.get("max_completion_tokens")) == MAX_TOKENS


def test_groq_profiles_pin_their_sampling() -> None:
    llama = profile_for("groq/llama-3.3-70b-versatile")
    gpt_oss = profile_for("groq/openai/gpt-oss-20b")
    assert (llama.context_window, llama.thinking) == (131_072, "n/a")
    assert (gpt_oss.context_window, gpt_oss.thinking) == (131_072, "on")
    assert llama.generate == {"temperature": 0.6, "top_p": 0.9, "seed": 0}
    assert gpt_oss.generate == {
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 0,
        "reasoning_effort": "medium",
    }


def test_an_unknown_model_lists_the_known_profiles() -> None:
    with pytest.raises(
        ValueError, match=re.escape("known profiles: ollama/redline-qwen3.5-0.8b, ")
    ):
        profile_for("ollama/qwen3.5:0.8b")


def test_no_profile_outside_an_eval() -> None:
    with pytest.raises(RuntimeError, match="no model is being evaluated"):
        active_profile()


def test_parse_modelfile() -> None:
    text = "# comment\nFROM qwen3.5:2b\nPARAMETER num_ctx 8192\nPARAMETER stop a b\n"
    assert parse_modelfile(text) == {"FROM": "qwen3.5:2b", "num_ctx": "8192"}
