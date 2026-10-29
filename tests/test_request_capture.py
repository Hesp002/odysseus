"""Fork: opt-in in-memory capture of model request bodies."""

import asyncio
import contextvars
import json

import httpx

from src import request_capture as rc


def _fresh(fn):
    """Run fn in a clean context with an empty buffer."""
    rc.clear()
    return contextvars.Context().run(fn)


def test_nothing_is_recorded_unless_the_turn_opted_in():
    def go():
        rc.record("http://h:11434/v1/chat/completions", {"model": "m"})
        return rc.records_for("s1")
    assert _fresh(go) == []


def test_records_model_requests_for_the_session_only():
    def go():
        rc.begin_turn("s1")
        rc.record("http://h:11434/v1/chat/completions", {"model": "m", "messages": []}, source="x")
        rc.record("http://h:8080/search?q=1", {"q": 1})  # not a model API
        return rc.records_for("s1"), rc.records_for("other")
    mine, other = _fresh(go)
    assert len(mine) == 1 and other == []
    assert json.loads(mine[0]["body"])["model"] == "m"
    assert mine[0]["source"] == "x"


def test_ring_buffer_keeps_only_the_newest():
    def go():
        rc.begin_turn("s1")
        for i in range(rc.MAX_RECORDS + 5):
            rc.record("http://h/api/generate", {"i": i})
        return rc.records_for("s1")
    rows = _fresh(go)
    assert len(rows) == rc.MAX_RECORDS
    assert json.loads(rows[0]["body"])["i"] == rc.MAX_RECORDS + 4  # newest first


def test_background_tasks_inherit_the_turn():
    async def main():
        rc.begin_turn("s1")

        async def background():
            rc.record("http://h/api/chat", {"bg": True})
        await asyncio.create_task(background())
        return rc.records_for("s1")
    rows = _fresh(lambda: asyncio.run(main()))
    assert len(rows) == 1


def test_httpx_hook_captures_exact_bytes_sent():
    payload = {"model": "m", "messages": [{"role": "user", "content": "héllo"}]}

    async def main():
        rc.begin_turn("s1")
        req = httpx.Request("POST", "http://h:11434/v1/chat/completions", json=payload)
        await rc.httpx_request_hook(req)
        return req.content, rc.records_for("s1")
    sent, rows = _fresh(lambda: asyncio.run(main()))
    assert rows[0]["body"] == sent.decode()


def test_clear_one_session():
    def go():
        rc.begin_turn("s1")
        rc.record("http://h/api/chat", {})
        rc.begin_turn("s2")
        rc.record("http://h/api/chat", {})
        rc.clear("s1")
        return rc.records_for("s1"), rc.records_for("s2")
    s1, s2 = _fresh(go)
    assert s1 == [] and len(s2) == 1
