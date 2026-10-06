"""A non-stream Ollama reply that ended without a finish signal is a failure.

When llama-server dies mid-generation (Vulkan "device lost" after another
process hung the GPU), Ollama drops the runner's error event and answers
HTTP 200 with the partial text and no finish_reason. Deep Research saved such
a half-written report as complete.
"""
import asyncio
import json

import pytest
from fastapi import HTTPException

import src.llm_core as llm_core
from src.deep_research import DeepResearcher

OLLAMA_V1 = "http://host.docker.internal:11434/v1/chat/completions"
OLLAMA_NATIVE = "http://localhost:11434/api/chat"


class _FakeResponse:
    is_success = True
    status_code = 200
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _v1_payload(finish_reason):
    return {"choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "a kabuki play play"},
        "finish_reason": finish_reason,
    }]}


def _call_async(monkeypatch, url, payload, model="qwen3:30b-a3b"):
    async def fake_post(client, url, headers, **kwargs):
        return _FakeResponse(payload)

    monkeypatch.setattr(llm_core, "httpx_post_kimi_aware_async", fake_post)
    llm_core._response_cache.clear()
    return asyncio.run(llm_core.llm_call_async(
        url, model, [{"role": "user", "content": "q"}], max_tokens=4096,
    ))


def _call_sync(monkeypatch, url, payload):
    monkeypatch.setattr(llm_core, "httpx_post_kimi_aware", lambda *a, **k: _FakeResponse(payload))
    llm_core._response_cache.clear()
    return llm_core.llm_call(url, "qwen3:30b-a3b", [{"role": "user", "content": "q"}], max_tokens=4096)


def test_async_ollama_v1_without_finish_reason_raises(monkeypatch):
    with pytest.raises(HTTPException) as exc:
        _call_async(monkeypatch, OLLAMA_V1, _v1_payload(None))
    assert exc.value.status_code == 502
    assert "stopped mid-response" in exc.value.detail
    assert not llm_core._response_cache


@pytest.mark.parametrize("reason", ["stop", "length", "tool_calls"])
def test_async_ollama_v1_with_finish_reason_passes(monkeypatch, reason):
    assert _call_async(monkeypatch, OLLAMA_V1, _v1_payload(reason)) == "a kabuki play play"


def test_async_ollama_native_not_done_raises(monkeypatch):
    payload = {"message": {"role": "assistant", "content": "partial"}, "done": False}
    with pytest.raises(HTTPException):
        _call_async(monkeypatch, OLLAMA_NATIVE, payload)


def test_async_ollama_native_done_passes(monkeypatch):
    payload = {"message": {"role": "assistant", "content": "full"}, "done": True, "done_reason": "stop"}
    assert _call_async(monkeypatch, OLLAMA_NATIVE, payload) == "full"


def test_async_non_ollama_missing_finish_reason_untouched(monkeypatch):
    # Other OpenAI-compatible servers may omit finish_reason; leave them alone.
    out = _call_async(monkeypatch, "http://llm.test/v1/chat/completions", _v1_payload(None), model="some-model")
    assert out == "a kabuki play play"


def test_async_other_local_server_missing_finish_reason_untouched(monkeypatch):
    # localhost counts as "Ollama" for URL shaping; the cut-off check must not.
    out = _call_async(monkeypatch, "http://localhost:8000/v1/chat/completions", _v1_payload(None), model="some-model")
    assert out == "a kabuki play play"


def test_sync_ollama_v1_without_finish_reason_raises(monkeypatch):
    with pytest.raises(HTTPException) as exc:
        _call_sync(monkeypatch, OLLAMA_V1, _v1_payload(None))
    assert "stopped mid-response" in exc.value.detail


def test_sync_ollama_v1_with_finish_reason_passes(monkeypatch):
    assert _call_sync(monkeypatch, OLLAMA_V1, _v1_payload("stop")) == "a kabuki play play"


def test_final_report_falls_back_to_draft_when_cut_off(monkeypatch):
    researcher = DeepResearcher.__new__(DeepResearcher)
    researcher.category = None
    researcher.max_report_tokens = 4096
    researcher._progress = None

    async def cut_off(*args, **kwargs):
        raise HTTPException(502, "The model stopped mid-response without finishing")

    monkeypatch.setattr(researcher, "_llm", cut_off)
    draft = "## Draft\n\nComplete evolving report."
    assert asyncio.run(researcher._final_report("47 Ronin", draft)) == draft


# ---------------------------------------------------------------------------
# Streaming: Ollama ends a failed reply without the final chunk / [DONE].
# ---------------------------------------------------------------------------

class _FakeStreamResp:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b""


class _FakeStreamCtx:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return _FakeStreamResp(self._lines)

    async def __aexit__(self, *a):
        return False


class _FakeStreamClient:
    def __init__(self, lines):
        self._lines = lines

    def stream(self, method, url, **kw):
        return _FakeStreamCtx(self._lines)


def _stream_text(monkeypatch, url, lines, model="qwen3:30b-a3b"):
    monkeypatch.setattr(llm_core, "_get_http_client", lambda: _FakeStreamClient(lines))
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda u: False)
    monkeypatch.setattr(llm_core, "note_model_activity", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "_clear_host_dead", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "get_context_length", lambda u, m: 32768)

    async def run():
        return [c async for c in llm_core.stream_llm(url, model, [{"role": "user", "content": "hi"}])]

    events = asyncio.run(run())
    text = ""
    for event in events:
        if event.startswith("data: {"):
            data = json.loads(event[len("data: "):])
            if not data.get("thinking"):
                text += data.get("delta") or ""
    assert events[-1] == "data: [DONE]\n\n"
    return text


def _v1_chunk(content, finish_reason=None):
    return "data: " + json.dumps({"choices": [{"delta": {"content": content}, "finish_reason": finish_reason}]})


def test_stream_ollama_v1_without_done_appends_warning(monkeypatch):
    text = _stream_text(monkeypatch, OLLAMA_V1, [_v1_chunk("a kabuki play"), _v1_chunk(" play")])
    assert text.startswith("a kabuki play play")
    assert text.endswith(llm_core._OLLAMA_STREAM_CUT_OFF_NOTE)


def test_stream_ollama_v1_finished_has_no_warning(monkeypatch):
    lines = [_v1_chunk("all done"), _v1_chunk("", "stop"), "data: [DONE]"]
    assert _stream_text(monkeypatch, OLLAMA_V1, lines) == "all done"


def test_stream_other_local_server_without_done_has_no_warning(monkeypatch):
    lines = [_v1_chunk("no done marker")]
    out = _stream_text(monkeypatch, "http://localhost:8000/v1/chat/completions", lines, model="some-model")
    assert out == "no done marker"


def test_stream_ollama_native_without_done_appends_warning(monkeypatch):
    lines = [json.dumps({"message": {"role": "assistant", "content": "partial"}, "done": False})]
    text = _stream_text(monkeypatch, OLLAMA_NATIVE, lines)
    assert text == "partial" + llm_core._OLLAMA_STREAM_CUT_OFF_NOTE


def test_stream_ollama_native_done_has_no_warning(monkeypatch):
    lines = [
        json.dumps({"message": {"role": "assistant", "content": "full"}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}),
    ]
    assert _stream_text(monkeypatch, OLLAMA_NATIVE, lines) == "full"
