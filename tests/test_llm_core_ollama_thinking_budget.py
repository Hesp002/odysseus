"""Regression guard: small-budget calls to thinking models on Ollama /v1 came back empty.

Ollama's OpenAI-compatible /v1 surface ignores ``"think": false`` for some
models (qwen3.5 on Ollama 0.33.x). Helper calls with tiny ``max_tokens``
(intent classification, titles, email triage) and the 128-token direct reply
path spent the entire budget on reasoning, returned empty content, and surfaced
as "[Agent stopped: Model request failed]". ``reasoning_effort: "none"`` is
honoured, so it is sent whenever the budget is too small to think in, while
unbounded chat/agent turns keep their reasoning.
"""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import pytest

from src.llm_core import _OLLAMA_MIN_THINKING_BUDGET, _apply_ollama_thinking_suppression

OLLAMA_V1 = "http://host.docker.internal:11434/v1/chat/completions"


@pytest.mark.parametrize("max_tokens", [5, 20, 128, 200, _OLLAMA_MIN_THINKING_BUDGET - 1])
def test_small_budget_disables_reasoning(max_tokens):
    payload = {}
    _apply_ollama_thinking_suppression(payload, OLLAMA_V1, "qwen3.5:9b", max_tokens)
    assert payload == {"think": False, "reasoning_effort": "none"}


@pytest.mark.parametrize("max_tokens", [None, 0, _OLLAMA_MIN_THINKING_BUDGET, 2048])
def test_unbounded_or_large_budget_keeps_reasoning(max_tokens):
    payload = {}
    _apply_ollama_thinking_suppression(payload, OLLAMA_V1, "qwen3.5:9b", max_tokens)
    assert payload == {"think": False}


@pytest.mark.parametrize(
    "url, model",
    [
        ("https://api.openai.com/v1/chat/completions", "qwen3.5:9b"),  # not local Ollama
        ("http://localhost:11434/api/chat", "qwen3.5:9b"),  # native API, not /v1
        (OLLAMA_V1, "llama3.1:8b"),  # not a thinking model
    ],
)
def test_other_endpoints_and_models_untouched(url, model):
    payload = {}
    _apply_ollama_thinking_suppression(payload, url, model, 20)
    assert payload == {}
