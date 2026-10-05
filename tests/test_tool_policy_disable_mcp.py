from src.tool_policy import build_effective_tool_policy


def test_policy_keeps_mcp_by_default():
    assert build_effective_tool_policy(last_user_message="hi").disable_mcp is False


def test_policy_can_drop_mcp_for_a_turn():
    policy = build_effective_tool_policy(
        last_user_message="My dog is named Ace. Remember that.",
        disable_mcp=True,
    )
    assert policy.disable_mcp is True
    assert not policy.blocks("manage_memory")
