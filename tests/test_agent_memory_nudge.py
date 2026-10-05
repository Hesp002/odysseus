"""Regression: an explicit "remember that..." turn that ends without a
manage_memory call gets one nudge to make the call.

Small local models reply "I've noted that Ace is your dog" without emitting
the tool block, so nothing is stored.
"""

import asyncio
import json

import src.agent_loop as al


_NUDGE_MARKER = "NOTHING was saved"


def _collect(gen):
    async def _run():
        return [c async for c in gen]
    return asyncio.run(_run())


def _events(chunks):
    out = []
    for c in chunks:
        if c.startswith("data: ") and not c.startswith("data: [DONE]"):
            try:
                out.append(json.loads(c[6:]))
            except Exception:
                pass
    return out


def _run_loop(monkeypatch, user_text, round_texts, max_rounds=5, **loop_kwargs):
    monkeypatch.setattr(al, "get_setting", lambda key, default=None: default, raising=False)
    monkeypatch.setattr(al, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(al, "estimate_tokens", lambda *a, **k: 10, raising=False)
    # Run as the admin/single user; non-admins can't use manage_memory at all.
    monkeypatch.setattr(al, "blocked_tools_for_owner", lambda owner: set(), raising=False)

    executed = []

    async def _fake_exec(block, *a, **k):
        executed.append(block.tool_type)
        return (block.tool_type, {"output": "ok", "exit_code": 0})
    monkeypatch.setattr(al, "execute_tool_block", _fake_exec, raising=False)

    calls = []

    async def _fake_stream(_candidates, messages, **kwargs):
        calls.append([dict(m) for m in messages])
        text = round_texts[min(len(calls), len(round_texts)) - 1]
        yield f'data: {json.dumps({"delta": text})}\n\n'
        yield "data: [DONE]\n\n"
    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)

    gen = al.stream_agent_loop(
        "http://x/v1", "m",
        [{"role": "user", "content": user_text}],
        max_rounds=max_rounds,
        relevant_tools={"manage_memory"},
        **loop_kwargs,
    )
    events = _events(_collect(gen))
    return calls, executed, events


def _nudged(messages):
    return any(_NUDGE_MARKER in str(m.get("content", "")) for m in messages)


def test_claimed_save_without_tool_call_is_nudged_once(monkeypatch):
    calls, executed, _ = _run_loop(
        monkeypatch,
        "My dog is named Ace. Remember that.",
        ["Got it, I've noted that your dog is named Ace."],
    )
    # One nudge, then the turn ends even though the model still didn't call it.
    assert len(calls) == 2
    assert not _nudged(calls[0])
    assert _nudged(calls[1])
    assert executed == []


def test_nudge_leads_to_manage_memory_call(monkeypatch):
    calls, executed, _ = _run_loop(
        monkeypatch,
        "My dog is named Ace. Remember that.",
        [
            "Got it!",
            "```manage_memory\nadd\nThe user's dog is named Ace.\n```",
            "Saved: your dog is named Ace.",
        ],
    )
    assert executed == ["manage_memory"]
    assert len(calls) == 3


def test_no_nudge_when_manage_memory_called_first(monkeypatch):
    calls, executed, _ = _run_loop(
        monkeypatch,
        "Remember that I'm allergic to cats",
        [
            "```manage_memory\nadd\nThe user is allergic to cats.\n```",
            "Saved: you're allergic to cats.",
        ],
    )
    assert executed == ["manage_memory"]
    assert len(calls) == 2
    assert not any(_nudged(c) for c in calls)


def test_no_nudge_for_recall_question(monkeypatch):
    calls, executed, _ = _run_loop(
        monkeypatch,
        "Do you remember what my dog's name is?",
        ["I don't have that saved yet."],
    )
    assert len(calls) == 1
    assert executed == []


def test_no_second_save_after_approved_manage_memory(monkeypatch):
    # The add needed approval; the resumed run executes the approved call and
    # the model just confirms. That must not trigger a nudge (duplicate add).
    from src.tool_approvals import ToolApprovalStore
    from src.tool_capabilities import capabilities_for_action

    content = "add\nThe user's dog is named Ace."
    store = ToolApprovalStore()
    pending = store.create(
        owner="alice",
        session_id="session-1",
        origin_run_id="run-1",
        tool_name="manage_memory",
        content=content,
        workspace=None,
        external_untrusted_context_seen=True,
        capabilities=capabilities_for_action("manage_memory", content),
    )
    grant = store.consume(
        pending.approval_id, decision="approve", owner="alice", session_id="session-1",
    )

    calls, executed, _ = _run_loop(
        monkeypatch,
        "My dog is named Ace. Remember that.",
        ["Ace is now part of your memory."],
        owner="alice",
        session_id="session-1",
        exact_approval=grant,
    )
    assert executed == ["manage_memory"]
    assert len(calls) == 1
    assert not _nudged(calls[0])
