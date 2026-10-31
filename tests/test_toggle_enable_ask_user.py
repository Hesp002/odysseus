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
    _, result = _ask({"enable": "email"})
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


def test_bare_question_gets_yes_no_options():
    _, result = _ask({"question": "Enable web search by selecting 'Yes'?"})
    assert [o["label"] for o in result["ask_user"]["options"]] == ["Yes", "No"]


def test_web_off_marker_is_detected_only_from_the_note():
    from src.agent_tools.interaction_tools import toggle_off_note, web_toggle_off

    assert web_toggle_off([{"role": "system", "content": "Base prompt.\n" + toggle_off_note(["shell", "web"])}])
    assert not web_toggle_off([{"role": "system", "content": toggle_off_note(["shell"])}])
    assert not web_toggle_off([{"role": "user", "content": toggle_off_note(["web"])}])


def test_backstop_turns_web_question_into_enable_card():
    from src.agent_tools.interaction_tools import apply_web_off_backstop

    card = apply_web_off_backstop({"question": "Enable web search to get the latest F1 winner?",
                                   "options": [{"label": "Yes"}, {"label": "No"}]})
    assert card["enable"] == "web"
    assert card["options"][0]["label"] == "Yes, search the web"
    assert card["options"][0]["value"] == "enable"


def test_backstop_adds_search_option_to_clarifying_question():
    from src.agent_tools.interaction_tools import apply_web_off_backstop

    options = [{"label": str(i)} for i in range(6)]
    card = apply_web_off_backstop({"question": "Is 'Omarchy' a typo?", "options": options})
    assert len(card["options"]) == 6
    assert card["options"][-1] == {"label": "Search the web instead",
                                   "description": "Turns on Web search", "value": "enable"}
    assert card["enable"] == "web"


def test_backstop_leaves_enable_and_approval_cards_alone():
    from src.agent_tools.interaction_tools import apply_web_off_backstop

    shell = {"question": "q", "enable": "shell", "options": []}
    approval = {"question": "q", "kind": "tool_approval", "options": [{"label": "Allow"}, {"label": "Deny"}]}
    assert apply_web_off_backstop(shell) is shell
    assert apply_web_off_backstop(approval) is approval


def test_unknown_turn_needs_the_search_grant_to_run_web_search():
    # The contract offered web_search for "unknown", but the request
    # authority (built from the same classifier) had no grant, so the call
    # was denied at execution. chat_routes now passes search_browser.
    from src.agent_runtime.authority import create_request_authority

    message = "Check the price of the framework desktop."
    assert requested_capabilities(message, []) == {"unknown"}
    granted = lambda auth: {g.tool for g in auth.grants}
    assert "web_search" not in granted(create_request_authority(message, owner="u"))
    authority = create_request_authority(message, owner="u", capabilities={"search_browser"})
    assert {"web_search", "web_fetch"} <= granted(authority)


def test_unclassified_turn_needs_the_search_grant_to_run_web_search():
    # No family at all (not even "unknown"), and the semantic fallback skips
    # the turn when its model is cold. With Web on, chat_routes grants
    # search_browser for any result <= {"unknown"}, or every search the
    # model makes is "outside server request authority".
    from src.agent_runtime.authority import create_request_authority

    message = "How old was Patrick Stewart when he died, and what was the cause?"
    assert requested_capabilities(message, []) <= frozenset({"unknown"})
    granted = lambda auth: {g.tool for g in auth.grants}
    assert "web_search" not in granted(create_request_authority(message, owner="u"))
    authority = create_request_authority(message, owner="u", capabilities={"search_browser"})
    assert {"web_search", "web_fetch"} <= granted(authority)
