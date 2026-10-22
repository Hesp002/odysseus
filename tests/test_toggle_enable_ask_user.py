"""Fork: unclassified lookups get web tools; toggles that are off can be
turned back on from an ask_user Yes/No card."""
import asyncio
import json

from src.agent_tools.interaction_tools import AskUserTool, ENABLE_OPTIONS
from src.tool_policy import ToolPolicy, WEB_TOOL_NAMES
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
from src.turn_contract import requested_capabilities, resolve_turn_contract


def _ask(args):
    return asyncio.run(AskUserTool().execute(json.dumps(args), None))


def test_enable_web_returns_fixed_yes_no_card():
    desc, result = _ask({"question": "Web search is off. Turn it on?", "enable": "web"})
    card = result["ask_user"]
    assert result["exit_code"] == 0
    assert card["enable"] == "web"
    assert card["multi"] is False
    assert [o.get("value") for o in card["options"]] == ["enable", None]
    assert desc == "ask_user: enable web"


def test_enable_fills_default_question_and_ignores_model_options():
    _, result = _ask({"enable": "shell", "options": [{"label": "Maybe"}, {"label": "Later"}]})
    card = result["ask_user"]
    assert card["question"] == "Shell access is off. Turn it on for this?"
    assert card["options"][0]["label"] == "Yes, allow shell access"


def test_unknown_enable_value_is_a_normal_ask_user():
    _, result = _ask({"question": "Pick one", "enable": "email",
                      "options": [{"label": "A"}, {"label": "B"}]})
    assert "enable" not in result["ask_user"]
    _, result = _ask({"question": "Pick one", "enable": "email"})
    assert result["exit_code"] == 1


def test_schema_documents_enable():
    schema = next(s for s in FUNCTION_TOOL_SCHEMAS if s["function"]["name"] == "ask_user")
    params = schema["function"]["parameters"]
    assert params["properties"]["enable"]["enum"] == ["web", "shell"]
    assert params["required"] == ["question"]


def test_yes_labels_route_the_follow_up_turn():
    history = [
        {"role": "user", "content": "Check the price of the framework desktop."},
        {"role": "assistant", "content": "Web search is off. Turn it on for this?"},
    ]
    assert "search_browser" in requested_capabilities(ENABLE_OPTIONS["web"][0]["label"], history)
    assert "shell_files" in requested_capabilities(ENABLE_OPTIONS["shell"][0]["label"], history)


def _contract(policy):
    # What chat_routes passes for an "unknown" turn after the fork's remap.
    return resolve_turn_contract(
        capabilities=frozenset({"search_browser"}), schemas=FUNCTION_TOOL_SCHEMAS,
        policy=policy, required_capabilities=frozenset(),
    )


def test_unknown_turn_with_web_on_offers_web_search():
    contract = _contract(ToolPolicy())
    assert {"web_search", "web_fetch", "ask_user"} <= set(contract.offered)
    assert not contract.unavailable


def test_unknown_turn_with_web_off_still_offers_ask_user():
    contract = _contract(ToolPolicy(disabled_tools=frozenset(WEB_TOOL_NAMES | {"private_browser"})))
    assert "ask_user" in contract.offered
    assert not set(contract.offered) & WEB_TOOL_NAMES
    assert not contract.unavailable
