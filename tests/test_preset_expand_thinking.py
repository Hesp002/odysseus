"""Regression: "Expand with AI" put the model's reasoning into the persona box.

With max_tokens=500 the Ollama /v1 path suppresses thinking, and qwen3 then
reasons out loud in the answer and runs out of budget. The endpoint now gives
the model room to think separately and strips any leaked <think> block.
"""

import asyncio
from unittest.mock import MagicMock

import routes.preset_routes as preset_routes
from src.llm_core import _OLLAMA_MIN_THINKING_BUDGET


class _Req:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def test_expand_gives_room_to_think_and_strips_thinking(monkeypatch):
    import src.ai_interaction as ai
    import src.llm_core as llm_core

    seen = {}

    async def fake_llm(url, model, messages, **kwargs):
        seen["max_tokens"] = kwargs.get("max_tokens")
        return "<think>The user wants a Frasier prompt.</think>\nYou are Frasier Crane, a radio psychiatrist."

    monkeypatch.setattr(llm_core, "llm_call_async", fake_llm)
    monkeypatch.setattr(ai, "_resolve_model", lambda spec, owner=None: ("http://x:11434/v1", "qwen3:30b-a3b", {}))
    monkeypatch.setattr(preset_routes, "effective_user", lambda request: "kyle")

    router = preset_routes.setup_preset_routes(MagicMock())
    expand = next(r.endpoint for r in router.routes if r.path == "/api/presets/expand")
    result = asyncio.run(expand(_Req({"name": "Frasier Crane", "prompt": "radio psychiatrist"})))

    assert seen["max_tokens"] >= _OLLAMA_MIN_THINKING_BUDGET
    assert result == {"success": True, "prompt": "You are Frasier Crane, a radio psychiatrist."}
