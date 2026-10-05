"""Regression: the compact "native tool calling" prompt is only used when tool
schemas are actually sent.

Local Ollama /v1 without supports_tools gets no schemas (fenced tool blocks
instead), but used to get the compact prompt anyway: "use native tool calls;
do not write tool syntax" plus bare tool names. qwen3:30b-a3b then invented
`manage_memory({"fact": ...})`, which nothing parses, so tools never ran.
"""

import asyncio
import json

import src.agent_loop as al


_COMPACT_MARKER = "native tool/function calling"


def _system_prompt_for(monkeypatch, endpoint_url, model):
    monkeypatch.setattr(al, "get_setting", lambda key, default=None: default, raising=False)
    monkeypatch.setattr(al, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(al, "estimate_tokens", lambda *a, **k: 10, raising=False)
    monkeypatch.setattr(al, "blocked_tools_for_owner", lambda owner: set(), raising=False)
    # Pin the transport decision so the test doesn't depend on a database.
    is_ollama = ":11434/" in endpoint_url
    monkeypatch.setattr(
        al,
        "_agent_route_tool_mode",
        lambda url, m, owner=None, headers=None: (not is_ollama, False, is_ollama),
        raising=False,
    )
    al._cached_base_prompt = None
    al._cached_base_prompt_key = None

    seen = {}

    async def _fake_stream(_candidates, messages, **kwargs):
        seen.setdefault("messages", [dict(m) for m in messages])
        seen.setdefault("tools", kwargs.get("tools"))
        yield f'data: {json.dumps({"delta": "ok"})}\n\n'
        yield "data: [DONE]\n\n"
    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)

    async def _run():
        gen = al.stream_agent_loop(
            endpoint_url, model,
            [{"role": "user", "content": "My dog is named Ace. Remember that."}],
            max_rounds=1,
            relevant_tools={"manage_memory"},
        )
        return [c async for c in gen]
    asyncio.run(_run())
    return "\n".join(
        str(m.get("content", "")) for m in seen["messages"] if m.get("role") == "system"
    )


def test_ollama_without_native_tools_gets_fenced_block_prompt(monkeypatch):
    prompt = _system_prompt_for(monkeypatch, "http://host:11434/v1", "qwen3:30b-a3b")
    assert _COMPACT_MARKER not in prompt
    assert "```manage_memory" in prompt


def test_api_model_with_schemas_keeps_compact_prompt(monkeypatch):
    prompt = _system_prompt_for(monkeypatch, "https://api.openai.com/v1", "gpt-5")
    assert _COMPACT_MARKER in prompt
