"""Chat personas: CRUD, per-chat binding, prompt override, and memory scoping.

A persona is a CrewMember (not the personal assistant). A chat created with a
persona answers as that persona and only sees/writes that persona's memories;
Odysseus chats only see untagged memories.
"""

import asyncio
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.personas as personas
from core.database import Base, CrewMember
from core.database import Session as DbSession
from src.memory import MemoryManager


@pytest.fixture
def db(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(personas, "SessionLocal", TestSessionLocal)
    return TestSessionLocal


def _add_session(db, sid, owner="kyle", persona_id=None):
    s = db()
    s.add(DbSession(id=sid, name=sid, endpoint_url="", model="m", owner=owner,
                    crew_member_id=persona_id))
    s.commit()
    s.close()


# ── CRUD and session binding ─────────────────────────────────────────

def test_persona_crud_excludes_personal_assistant(db):
    s = db()
    s.add(CrewMember(id="assistant-1", owner="kyle", name="Assistant", is_default_assistant=True))
    s.commit()
    s.close()

    frasier = personas.create_persona("kyle", "Frasier Crane", "A pompous psychiatrist.")
    assert [p["name"] for p in personas.list_personas("kyle")] == ["Frasier Crane"]
    assert personas.get_persona("assistant-1", "kyle") is None
    assert personas.list_personas("someone-else") == []

    updated = personas.update_persona(frasier["id"], "kyle", personality="Radio psychiatrist.")
    assert updated["personality"] == "Radio psychiatrist."
    assert personas.update_persona(frasier["id"], "someone-else", name="x") is None


def test_session_persona_and_delete_unlinks_chats(db):
    marcus = personas.create_persona("kyle", "Marcus Aurelius", "Stoic emperor.")
    _add_session(db, "chat-marcus", persona_id=marcus["id"])
    _add_session(db, "chat-plain")

    assert personas.session_persona("chat-marcus")["name"] == "Marcus Aurelius"
    assert personas.session_persona("chat-plain") is None
    assert personas.session_persona(None) is None

    assert personas.delete_persona(marcus["id"], "kyle")
    assert personas.session_persona("chat-marcus") is None
    s = db()
    assert s.query(DbSession).filter(DbSession.id == "chat-marcus").first().crew_member_id is None
    s.close()


def test_persona_system_prompt_names_the_character():
    prompt = personas.persona_system_prompt({"name": "Frasier Crane", "personality": "A psychiatrist."})
    assert prompt.startswith("Your name is Frasier Crane. A psychiatrist.")
    assert "Stay in character" in prompt
    assert personas.persona_system_prompt(None) is None


def test_apply_session_persona_overrides_character_keeps_task_preset(monkeypatch):
    from routes.chat_helpers import PresetInfo, apply_session_persona

    persona = {"id": "p1", "name": "Frasier Crane", "personality": "A psychiatrist."}
    monkeypatch.setattr(personas, "session_persona", lambda sid: persona if sid == "s1" else None)

    plain = PresetInfo(temperature=1.0, max_tokens=0, system_prompt=None, character_name="")
    assert apply_session_persona(plain, "s2") is plain

    out = apply_session_persona(plain, "s1")
    assert out.character_name == "Frasier Crane"
    assert out.system_prompt.startswith("Your name is Frasier Crane.")

    task = PresetInfo(temperature=0.2, max_tokens=8000, system_prompt="Analyze code.", character_name="")
    out = apply_session_persona(task, "s1")
    assert out.system_prompt.endswith("Analyze code.")
    assert out.temperature == 0.2

    character = PresetInfo(temperature=1.0, max_tokens=0, system_prompt="Your name is Bob.", character_name="Bob")
    out = apply_session_persona(character, "s1")
    assert "Bob" not in out.system_prompt
    assert out.character_name == "Frasier Crane"


# ── Memory scoping ───────────────────────────────────────────────────

def test_filter_for_persona():
    entries = [{"id": "a"}, {"id": "b", "persona": "p1"}, {"id": "c", "persona": "p2"}]
    assert [e["id"] for e in personas.filter_for_persona(entries, None)] == ["a"]
    assert [e["id"] for e in personas.filter_for_persona(entries, "p1")] == ["b"]


@pytest.fixture
def ai_memory(tmp_path, monkeypatch):
    import src.ai_interaction as ai

    manager = MemoryManager(str(tmp_path))
    monkeypatch.setattr(ai, "_memory_manager", manager, raising=False)
    monkeypatch.setattr(ai, "_memory_vector", None, raising=False)
    monkeypatch.setattr(
        personas, "session_persona_id",
        lambda sid: {"frasier-chat": "p-frasier"}.get(sid),
    )
    return ai, manager


def test_manage_memory_add_and_list_are_scoped_to_the_chat_persona(ai_memory):
    ai, manager = ai_memory
    run = lambda content, sid: asyncio.run(ai.do_manage_memory(content, session_id=sid, owner="kyle"))

    run("add\nThe user's dog is named Ace.", "odysseus-chat")
    run("add\nThe user listens to KACL.", "frasier-chat")

    stored = {m["text"]: m.get("persona") for m in manager.load(owner="kyle")}
    assert stored == {
        "The user's dog is named Ace.": None,
        "The user listens to KACL.": "p-frasier",
    }

    frasier_list = run("list", "frasier-chat")["results"]
    assert "KACL" in frasier_list and "Ace" not in frasier_list
    odysseus_list = run("list", "odysseus-chat")["results"]
    assert "Ace" in odysseus_list and "KACL" not in odysseus_list

    frasier_search = run("search\nAce", "frasier-chat")
    assert "dog is named" not in json.dumps(frasier_search)
    assert "dog is named" in json.dumps(run("search\nAce", "odysseus-chat"))


def test_manage_memory_cannot_edit_or_delete_another_personas_memory(ai_memory):
    ai, manager = ai_memory
    run = lambda content, sid: asyncio.run(ai.do_manage_memory(content, session_id=sid, owner="kyle"))
    added = run("add\nThe user's dog is named Ace.", "odysseus-chat")

    assert "error" in run(f"delete\n{added['memory_id']}", "frasier-chat")
    assert "error" in run(f"edit\n{added['memory_id']}\nchanged", "frasier-chat")
    assert [m["text"] for m in manager.load(owner="kyle")] == ["The user's dog is named Ace."]


def test_chat_context_only_injects_the_personas_memories(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from src.chat_processor import ChatProcessor

    manager = MemoryManager(str(tmp_path))
    entries = [
        manager.add_entry("The user's dog is named Ace.", owner="kyle"),
        personas.tag_entry(manager.add_entry("The user listens to KACL.", owner="kyle"), "p-frasier"),
    ]
    for e in entries:
        e["pinned"] = True
    manager.save(entries)
    monkeypatch.setattr(
        personas, "session_persona_id",
        lambda sid: {"frasier-chat": "p-frasier"}.get(sid),
    )

    processor = ChatProcessor.__new__(ChatProcessor)
    processor.memory_manager = manager
    processor._select_pinned_memories = lambda message, pinned: pinned

    def used(session_id):
        processor._last_used_memories = []
        try:
            processor.build_context_preface(
                "hello", SimpleNamespace(id=session_id), use_rag=False, use_web=False,
                owner="kyle",
            )
        except Exception:
            pass  # only the memory step matters; later steps need more wiring
        return [m["text"] for m in processor._last_used_memories]

    assert used("frasier-chat") == ["The user listens to KACL."]
    assert used("odysseus-chat") == ["The user's dog is named Ace."]


def test_remove_persona_memories(tmp_path):
    manager = MemoryManager(str(tmp_path))
    entries = [
        manager.add_entry("Odysseus fact", owner="kyle"),
        personas.tag_entry(manager.add_entry("Frasier fact", owner="kyle"), "p-frasier"),
    ]
    manager.save(entries)
    assert personas.remove_persona_memories(manager, None, "p-frasier", "kyle") == 1
    assert [m["text"] for m in manager.load(owner="kyle")] == ["Odysseus fact"]


def test_memory_audit_of_odysseus_keeps_persona_memories(tmp_path, monkeypatch):
    import src.llm_core as llm_core
    from services.memory import memory_extractor

    manager = MemoryManager(str(tmp_path))
    odysseus = [manager.add_entry(f"Odysseus fact {i}", owner="kyle") for i in range(3)]
    frasier = personas.tag_entry(manager.add_entry("Frasier fact", owner="kyle"), "p-frasier")
    manager.save(odysseus + [frasier])

    seen = {}

    async def fake_llm(*args, **kwargs):
        messages = kwargs.get("messages") or (args[2] if len(args) > 2 else [])
        seen["prompt"] = json.dumps(messages)
        return json.dumps([{"id": e["id"], "text": e["text"]} for e in odysseus[:2]])
    monkeypatch.setattr(llm_core, "llm_call_async", fake_llm)

    asyncio.run(memory_extractor.audit_memories(manager, None, "http://x/v1", "m", owner="kyle"))

    assert "Frasier fact" not in seen.get("prompt", "")
    texts = sorted(m["text"] for m in manager.load(owner="kyle"))
    assert texts == ["Frasier fact", "Odysseus fact 0", "Odysseus fact 1"]


def test_persona_with_a_single_memory_still_recalls_it(tmp_path):
    # Regression: IDF over a one-memory slice gave "name" ~zero weight, so
    # Marcus never saw "My name is Kyle Hespe" when asked about the name.
    from src.chat_processor import ChatProcessor

    manager = MemoryManager(str(tmp_path))
    pool = [
        manager.add_entry("My name is Kyle Hespe, can be called Kyle", owner="kyle"),
        manager.add_entry("Kyle Hespe's dog is named Ace.", owner="kyle"),
        manager.add_entry("Kyle runs a three-node Proxmox cluster.", owner="kyle"),
        personas.tag_entry(manager.add_entry("My name is Kyle Hespe", owner="kyle"), "p-marcus"),
    ]
    processor = ChatProcessor.__new__(ChatProcessor)
    processor.memory_manager = manager
    processor.memory_vector = None
    marcus = personas.filter_for_persona(pool, "p-marcus")

    hits = processor._hybrid_retrieve("You know my name, don't you?", marcus, k=5, idf_corpus=pool)
    assert [m["text"] for m in hits] == ["My name is Kyle Hespe"]
    assert processor._hybrid_retrieve("What is my dog called?", marcus, k=5, idf_corpus=pool) == []


def test_persona_chat_preprocessing_skips_link_fetching(monkeypatch):
    import src.chat_handler as ch

    seen = []
    monkeypatch.setattr(ch, "extract_urls", lambda text: seen.append(text) or [])
    monkeypatch.setattr(personas, "session_persona_id", lambda sid: "p1" if sid == "persona-chat" else None)
    handler = ch.ChatHandler.__new__(ch.ChatHandler)

    def run(sid):
        from types import SimpleNamespace
        try:
            asyncio.run(handler.preprocess_message(
                "look at https://example.com", [], SimpleNamespace(id=sid, history=[]),
            ))
        except Exception:
            pass  # only the URL step matters here

    run("persona-chat")
    assert seen == []
    run("odysseus-chat")
    assert seen == ["look at https://example.com"]


def test_string_false_use_web_does_not_trigger_chat_web_search():
    # Regression: a persona chat passed use_web="false"; build_chat_context
    # used the raw string as truthy and ran a Brave search.
    from routes.chat_helpers import _tool_toggle_enabled

    assert _tool_toggle_enabled("false") is False
    assert _tool_toggle_enabled(None) is False
    assert _tool_toggle_enabled("true") is True
    assert _tool_toggle_enabled(True) is True


def test_tidy_route_audits_one_persona_scope_at_a_time(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    import routes.memory.memory_routes as memory_routes

    manager = MemoryManager(str(tmp_path))
    manager.save([
        manager.add_entry("My name is Kyle Hespe", owner="kyle"),
        personas.tag_entry(manager.add_entry("My name is Kyle Hespe", owner="kyle"), "p-marcus"),
        personas.tag_entry(manager.add_entry("Kyle likes jazz", owner="kyle"), "p-frasier"),
    ])
    monkeypatch.setattr(memory_routes, "get_current_user", lambda request: "kyle")
    monkeypatch.setattr(memory_routes, "resolve_task_endpoint", lambda *a, **k: ("http://x/v1", "m", None))
    monkeypatch.setattr(personas, "get_persona", lambda pid, owner=None: {"id": pid} if pid else None)
    calls = []

    async def fake_audit(*args, **kwargs):
        calls.append(kwargs.get("persona"))
        return {"before": 1, "after": 1}
    monkeypatch.setattr(memory_routes, "audit_memories", fake_audit)

    router = memory_routes.setup_memory_routes(manager, MagicMock())
    audit = next(r.endpoint for r in router.routes if r.path == "/api/memory/audit")
    run = lambda persona: asyncio.run(audit(request=None, session=None, persona=persona))

    run("")
    assert calls == [None]
    calls.clear()
    run("p-marcus")
    assert calls == ["p-marcus"]
    calls.clear()
    result = run("__all__")
    assert calls == [None, "p-frasier", "p-marcus"]
    assert result["before"] == 3


def test_memory_audit_of_a_persona_leaves_odysseus_alone(tmp_path, monkeypatch):
    import src.llm_core as llm_core
    from services.memory import memory_extractor

    manager = MemoryManager(str(tmp_path))
    odysseus = manager.add_entry("My name is Kyle Hespe", owner="kyle")
    marcus = [
        personas.tag_entry(manager.add_entry("My name is Kyle Hespe", owner="kyle"), "p-marcus"),
        personas.tag_entry(manager.add_entry("Kyle is called Kyle Hespe", owner="kyle"), "p-marcus"),
    ]
    manager.save([odysseus] + marcus)
    seen = {}

    async def fake_llm(*args, **kwargs):
        seen["prompt"] = json.dumps(kwargs.get("messages") or args)
        return json.dumps([{"id": marcus[0]["id"], "text": marcus[0]["text"]}])  # merge Marcus's two
    monkeypatch.setattr(llm_core, "llm_call_async", fake_llm)

    asyncio.run(memory_extractor.audit_memories(
        manager, None, "http://x/v1", "m", owner="kyle", persona="p-marcus",
    ))

    assert odysseus["id"] not in seen["prompt"]
    after = manager.load(owner="kyle")
    assert sorted((m.get("persona") or "", m["text"]) for m in after) == [
        ("", "My name is Kyle Hespe"),
        ("p-marcus", "My name is Kyle Hespe"),
    ]
