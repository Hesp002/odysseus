"""Fork: semantic fallback router (src/semantic_route.py)."""

import math

import pytest

from src import semantic_route as sr


def _lp(p):
    return math.log(p)


def test_letter_probabilities_merges_spaced_tokens_and_ignores_non_letters():
    probs = sr.letter_probabilities([
        {"token": " B", "logprob": _lp(0.5)},
        {"token": "B", "logprob": _lp(0.2)},
        {"token": "A", "logprob": _lp(0.2)},
        {"token": "Okay", "logprob": _lp(0.1)},
    ])
    assert list(probs) == ["B", "A"]
    assert probs["B"] == pytest.approx(0.7 / 0.9)


def test_temperature_softens_overconfidence_without_changing_the_winner():
    raw = [{"token": "B", "logprob": _lp(0.98)}, {"token": "A", "logprob": _lp(0.02)}]
    sharp = sr.letter_probabilities(raw, 1.0)
    soft = sr.letter_probabilities(raw, 2.0)
    assert next(iter(soft)) == "B"
    assert soft["B"] < sharp["B"]


def test_decision_is_actionable_only_for_confident_tool_families():
    assert sr.decide({"D": 0.95, "A": 0.05}, 0.9).actionable
    assert not sr.decide({"D": 0.85, "A": 0.15}, 0.9).actionable
    assert not sr.decide({"A": 0.99}, 0.9).actionable  # chat
    assert not sr.decide({"G": 0.99}, 0.9).actionable  # other
    assert sr.decide({}, 0.9) is None


def test_router_never_offers_host_access():
    offered = set().union(*(o.families for o in sr.ROUTE_OPTIONS))
    assert "shell_files" not in offered
    assert offered <= {"search_browser", "memory", "email", "calendar", "notes", "tasks"}


def test_prompt_prefills_empty_think_block_and_includes_context():
    prompt = sr.build_prompt("did she reply?", previous="email Linda about Friday")
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:")
    assert "email Linda about Friday" in prompt


def test_router_is_off_by_default_and_for_non_chatml_models(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_SEMANTIC_ROUTER", raising=False)
    monkeypatch.setenv("ODYSSEUS_SEMANTIC_ROUTER_MODEL", "qwen3:30b-a3b")
    assert sr.router_model() is None
    monkeypatch.setenv("ODYSSEUS_SEMANTIC_ROUTER", "1")
    assert sr.router_model() == "qwen3:30b-a3b"
    monkeypatch.setenv("ODYSSEUS_SEMANTIC_ROUTER_MODEL", "gemma4:26b")
    assert sr.router_model() is None


def test_base_url_strips_openai_suffix(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_SEMANTIC_ROUTER_URL", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")
    assert sr._base_url() == "http://host.docker.internal:11434"


def test_previous_user_message_skips_assistant_rows():
    history = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "reply"}]
    assert sr.previous_user_message(history) == "first"
    assert sr.previous_user_message([]) is None


async def test_semantic_route_fails_open(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_SEMANTIC_ROUTER", "1")
    monkeypatch.setenv("ODYSSEUS_SEMANTIC_ROUTER_MODEL", "qwen3:30b-a3b")

    async def boom(*args, **kwargs):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(sr, "score_message", boom)
    assert await sr.semantic_route("check the price of x") is None


async def test_semantic_route_applies_configured_threshold(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_SEMANTIC_ROUTER", "1")
    monkeypatch.setenv("ODYSSEUS_SEMANTIC_ROUTER_MODEL", "qwen3:30b-a3b")
    monkeypatch.setenv("ODYSSEUS_SEMANTIC_ROUTER_THRESHOLD", "0.5")

    async def fake(*args, **kwargs):
        return {"B": 0.6, "A": 0.4}

    monkeypatch.setattr(sr, "score_message", fake)
    decision = await sr.semantic_route("check the price of x")
    assert decision.option.label == "web" and decision.actionable
    assert decision.option.families == frozenset({"search_browser"})


def test_routed_family_is_offered_and_granted_for_an_unplaced_turn():
    # What chat_routes does when the lexical contract finds nothing and the
    # router picks email: the family replaces the empty result and is
    # passed to the request authority as well.
    from src.agent_runtime.authority import create_request_authority
    from src.tool_policy import ToolPolicy
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    from src.turn_contract import requested_capabilities, resolve_turn_contract

    message = "did Linda reply to me yet"
    assert requested_capabilities(message, []) <= {"unknown"}
    families = sr.decide({"D": 0.97, "A": 0.03}, 0.9).option.families
    contract = resolve_turn_contract(
        capabilities=families, schemas=FUNCTION_TOOL_SCHEMAS,
        policy=ToolPolicy(), required_capabilities=families,
    )
    assert {"list_emails", "search_emails"} <= set(contract.offered)
    granted = {g.tool for g in create_request_authority(message, owner="u", capabilities=families).grants}
    assert {"list_emails", "search_emails"} <= granted
    assert "bash" not in granted


async def _route_with_load_state(monkeypatch, loaded):
    monkeypatch.setenv("ODYSSEUS_SEMANTIC_ROUTER", "1")
    monkeypatch.setenv("ODYSSEUS_SEMANTIC_ROUTER_MODEL", "qwen3:30b-a3b")
    seen = {}

    async def fake_loaded(model):
        return loaded

    async def fake_score(*args, timeout, **kwargs):
        seen["timeout"] = timeout
        return {"A": 1.0}

    monkeypatch.setattr(sr, "model_loaded", fake_loaded)
    monkeypatch.setattr(sr, "score_message", fake_score)
    await sr.semantic_route("hello")
    return seen["timeout"]


async def test_warm_model_uses_short_timeout(monkeypatch):
    assert await _route_with_load_state(monkeypatch, True) == 4.0


async def test_cold_model_waits_for_the_load(monkeypatch):
    # Giving up at 4 s made Ollama cancel the ~13 s load the reply needs.
    assert await _route_with_load_state(monkeypatch, False) == 30.0
