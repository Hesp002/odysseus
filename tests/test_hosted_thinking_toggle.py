"""Fork: Tinfoil's reasoning models get the per-chat thinking switch, new
chats start with thinking on, and "off" really disables DeepSeek V4 thinking
on Tinfoil instead of only hiding it."""
from src.llm_core import _apply_tinfoil_thinking_mode
from src.model_profiles import default_thinking_mode, supports_user_thinking_toggle

_PROXY = "http://odysseus-tinfoil:3301/v1"


def test_tinfoil_reasoning_models_get_the_switch():
    assert supports_user_thinking_toggle("deepseek-v4-1-flash")
    assert supports_user_thinking_toggle("glm-5-3")
    assert supports_user_thinking_toggle("qwen3:30b-a3b")
    # gpt-oss reasons through effort levels, not an on/off switch.
    assert not supports_user_thinking_toggle("gpt-oss-120b")
    assert not supports_user_thinking_toggle("selected-model")


def test_new_chats_start_with_thinking_on_when_switchable():
    assert default_thinking_mode("deepseek-v4-1-flash") == "on"
    assert default_thinking_mode("qwen3:30b-a3b") == "on"
    assert default_thinking_mode("gpt-oss-120b") == "off"
    assert default_thinking_mode("") == "off"


def test_tinfoil_deepseek_thinking_follows_the_switch():
    for mode, expected in (("on", True), ("off", False)):
        payload = {}
        _apply_tinfoil_thinking_mode(payload, _PROXY, "deepseek-v4-1-flash", mode)
        assert payload == {"chat_template_kwargs": {"thinking": expected}}
    payload = {"chat_template_kwargs": {"other": 1}}
    _apply_tinfoil_thinking_mode(payload, "https://inference.tinfoil.sh/v1", "deepseek-v4-1-flash", "off")
    assert payload == {"chat_template_kwargs": {"other": 1, "thinking": False}}


def test_other_hosts_models_and_unset_mode_are_untouched():
    cases = (
        ("https://openrouter.ai/api/v1", "deepseek-v4-1-flash", "off"),
        (_PROXY, "glm-5-3", "off"),
        (_PROXY, "gpt-oss-120b", "off"),
        (_PROXY, "deepseek-v4-1-flash", None),
    )
    for url, model, mode in cases:
        payload = {}
        _apply_tinfoil_thinking_mode(payload, url, model, mode)
        assert payload == {}, (url, model, mode)
