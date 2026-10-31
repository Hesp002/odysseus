"""Fork: an approved web_fetch the site refused continues the turn; a
dispatch rejection of the approved action still ends it."""
from src.agent_loop import _approved_fetch_failure_continues

_SITE_403 = {
    "error": "web_fetch: https://apnews.com/article/x: HTTP 403: HTTP 403 for https://apnews.com/article/x",
    "exit_code": 1,
}


def test_site_refusal_of_approved_fetch_continues():
    assert _approved_fetch_failure_continues("web_fetch", _SITE_403)


def test_blocked_or_reapproval_still_ends_the_turn():
    assert not _approved_fetch_failure_continues("web_fetch", {**_SITE_403, "blocked": True})
    assert not _approved_fetch_failure_continues("web_fetch", {**_SITE_403, "approval_required": True})


def test_other_tools_keep_the_sealed_failure():
    assert not _approved_fetch_failure_continues("host_shell", _SITE_403)
    assert not _approved_fetch_failure_continues("manage_notes", _SITE_403)
    assert not _approved_fetch_failure_continues("web_fetch", "HTTP 403")
