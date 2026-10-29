"""Fork: opt-in, in-memory capture of the exact request bodies sent to models.

The browser's "Raw requests" toggle (Settings → Appearance) adds
capture_requests=true to /api/chat_stream. That turn — and anything it starts
(agent rounds, background memory extraction, the semantic router) — then
records each outgoing model request body here, byte for byte as sent.

Records live only in this process: a bounded ring buffer, never written to
disk, gone on restart. With the toggle off nothing is recorded.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from urllib.parse import urlsplit

MAX_RECORDS = 30
# A single payload above this is truncated so one huge attachment can't pin
# a lot of memory.
MAX_BODY_CHARS = 2_000_000

# Request paths of model APIs (OpenAI-compatible, Ollama native, Anthropic,
# Responses). Anything else on the shared client is ignored.
_MODEL_PATH_SUFFIXES = (
    "/chat/completions", "/completions", "/api/chat", "/api/generate",
    "/messages", "/responses",
)


@dataclass(frozen=True)
class _Turn:
    session_id: str
    turn_id: str


_turn: ContextVar[_Turn | None] = ContextVar("request_capture_turn", default=None)
_records: deque[dict] = deque(maxlen=MAX_RECORDS)
_lock = Lock()


def begin_turn(session_id: str) -> str:
    """Capture model requests made from the current context for this turn."""
    turn_id = uuid.uuid4().hex[:12]
    _turn.set(_Turn(str(session_id or ""), turn_id))
    return turn_id


def is_model_url(url: str) -> bool:
    path = urlsplit(str(url)).path.rstrip("/")
    return path.endswith(_MODEL_PATH_SUFFIXES)


def record(url: str, body, *, source: str = "") -> None:
    """Store one outgoing request body if this context is capturing."""
    turn = _turn.get()
    if turn is None or not is_model_url(url):
        return
    if isinstance(body, (bytes, bytearray)):
        text = bytes(body).decode("utf-8", errors="replace")
    elif isinstance(body, str):
        text = body
    else:
        text = json.dumps(body, ensure_ascii=False)
    truncated = len(text) > MAX_BODY_CHARS
    entry = {
        "id": uuid.uuid4().hex[:12],
        "session_id": turn.session_id,
        "turn_id": turn.turn_id,
        "time": time.time(),
        "url": str(url),
        "source": source,
        "bytes": len(text.encode("utf-8")),
        "truncated": truncated,
        "body": text[:MAX_BODY_CHARS],
    }
    with _lock:
        _records.append(entry)


async def httpx_request_hook(request) -> None:
    """httpx event hook for the shared LLM client."""
    try:
        record(str(request.url), request.content, source="llm_core")
    except Exception:
        # Diagnostics must never break a model call.
        pass


def records_for(session_id: str) -> list[dict]:
    """Newest first, for one session."""
    with _lock:
        rows = [dict(r) for r in _records if r["session_id"] == str(session_id)]
    return rows[::-1]


def clear(session_id: str | None = None) -> None:
    with _lock:
        if session_id is None:
            _records.clear()
            return
        keep = [r for r in _records if r["session_id"] != str(session_id)]
        _records.clear()
        _records.extend(keep)
