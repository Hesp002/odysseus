import json
import logging
import re

logger = logging.getLogger(__name__)

ENABLE_QUESTIONS = {
    "web": ("web search", "Web search is off. Turn it on for this?"),
    "shell": ("shell access", "Shell access is off. Turn it on for this?"),
}
ENABLE_OPTIONS = {
    "web": (
        {"label": "Yes, search the web", "description": "Turns on Web search", "value": "enable"},
        {"label": "No, answer without it", "description": ""},
    ),
    "shell": (
        {"label": "Yes, allow shell access", "description": "Turns on Shell Access", "value": "enable"},
        {"label": "No, answer without it", "description": ""},
    ),
}
_SEARCH_INSTEAD = {"label": "Search the web instead", "description": "Turns on Web search", "value": "enable"}
_TOGGLE_NEEDS = {
    "web": ("Web search", "current or online information"),
    "shell": ("Shell access", "running a command on this computer"),
}
_WEB_QUESTION_RE = re.compile(r"\b(?:web|search|online|internet|look\s+(?:it\s+)?up|browse)\b", re.I)


def toggle_off_note(kinds):
    """System note for agent turns where the user turned Web/Shell off."""
    return "\n".join(
        f"{_TOGGLE_NEEDS[k][0]} is turned OFF by the user. If answering needs {_TOGGLE_NEEDS[k][1]}, "
        f"do not pretend to use it and do not guess: call ask_user with "
        f"enable=\"{k}\" and a one-line question. The app shows Yes/No buttons."
        for k in kinds
    )


def web_toggle_off(messages):
    marker = toggle_off_note(["web"]).split(".", 1)[0]
    return any(
        isinstance(m, dict) and m.get("role") == "system" and marker in str(m.get("content") or "")
        for m in messages or ()
    )


def apply_web_off_backstop(card):
    """With Web off, any question card also offers to turn Web on.

    Small models often ask a clarifying question ("is Omarchy a typo?") or
    word a Yes/No in prose instead of passing enable="web". Don't rely on
    them: a question about searching becomes the fixed enable card, and any
    other question gains a "Search the web instead" option.
    """
    if card.get("enable") or card.get("kind") == "tool_approval":
        return card
    card = dict(card, enable="web", multi=False)
    if _WEB_QUESTION_RE.search(str(card.get("question") or "")):
        card["options"] = [dict(o) for o in ENABLE_OPTIONS["web"]]
    else:
        card["options"] = list(card.get("options") or [])[:5] + [dict(_SEARCH_INSTEAD)]
    return card

class AskUserTool:
    async def execute(self, content, ctx):
        """
        ask_user: the agent poses a multiple-choice question to the user to get a
        decision/clarification. This is a pure UI-control marker — no subprocess,
        no filesystem. It returns an `ask_user` payload that the agent loop turns
        into an `ask_user` SSE event and then ENDS the turn, so the chat waits for
        the user's selection (their choice arrives as the next message).
        """
        question, options, multi = "", [], False
        raw = (content or "").strip()
        try:
            parsed = json.loads(raw) if raw else {}
        except (ValueError, TypeError):
            parsed = {}

        if isinstance(parsed, dict):
            question = str(parsed.get("question", "")).strip()
            multi = bool(parsed.get("multi") or parsed.get("multiSelect"))
            for opt in (parsed.get("options") or []):
                if isinstance(opt, dict):
                    label = str(opt.get("label", "")).strip()
                    descr = str(opt.get("description", "")).strip()
                elif isinstance(opt, str):
                    label, descr = opt.strip(), ""
                else:
                    continue
                if label:
                    options.append({"label": label, "description": descr})
        else:
            question = raw

        # Fork: a request to turn a toggle back on gets fixed options. The
        # first one carries value "enable", which makes the card flip the
        # toggle before sending; its label is worded so the follow-up turn
        # classifies as a web/shell request.
        enable = parsed.get("enable") if isinstance(parsed, dict) else None
        if enable in ENABLE_OPTIONS:
            label, default_question = ENABLE_QUESTIONS[enable]
            question = question or default_question
            result = {
                "ask_user": {
                    "question": question,
                    "options": [dict(o) for o in ENABLE_OPTIONS[enable]],
                    "multi": False,
                    "enable": enable,
                },
                "output": f"Asked the user to turn on {label}: {question}\nAwaiting their answer.",
                "exit_code": 0,
            }
            logger.info("Tool executed: ask_user enable=%s", enable)
            return f"ask_user: enable {enable}", result

        if question and not options:
            # Fork: a bare question is almost always a yes/no one; show it
            # rather than bouncing an error back to a small model.
            options = [{"label": "Yes", "description": ""}, {"label": "No", "description": ""}]

        if not question or len(options) < 2:
            return "ask_user: invalid", {
                "error": (
                    "ask_user needs a non-empty `question` and at least 2 `options` "
                    "(each an object with a `label`, optional `description`)."
                ),
                "exit_code": 1,
            }

        options = options[:6]  # keep the choice list sane
        desc = f"ask_user: {question[:80]}"
        labels = ", ".join(o["label"] for o in options)
        result = {
            "ask_user": {"question": question, "options": options, "multi": multi},
            "output": f"Asked the user: {question}\nOptions: {labels}\nAwaiting their selection.",
            "exit_code": 0,
        }
        logger.info("Tool executed: %s (%d options, multi=%s)", desc, len(options), multi)
        return desc, result

class UpdatePlanTool:
    async def execute(self, content, ctx):
        """
        update_plan: the agent writes back to the active plan — tick an item done
        or revise steps (e.g. when the user asks to change something). Pure UI
        marker: returns a `plan_update` payload the agent loop turns into a
        `plan_update` SSE event; the frontend replaces the stored plan and refreshes
        the docked plan window. Does NOT end the turn.
        """
        raw = (content or "").strip()
        plan = ""
        try:
            parsed = json.loads(raw) if raw else {}
        except (ValueError, TypeError):
            parsed = {}

        if isinstance(parsed, dict) and parsed.get("plan"):
            plan = str(parsed.get("plan", "")).strip()
        else:
            plan = raw

        if not plan:
            return "update_plan: invalid", {
                "error": "update_plan needs a non-empty `plan` (the full updated checklist as markdown).",
                "exit_code": 1,
            }

        plan = plan[:8192]
        done = plan.count("- [x]") + plan.count("- [X]")
        total = done + plan.count("- [ ]")
        desc = f"update_plan: {done}/{total} done" if total else "update_plan"
        result = {
            "plan_update": {"plan": plan},
            "output": f"Plan updated ({done}/{total} steps complete)." if total else "Plan updated.",
            "exit_code": 0,
        }
        logger.info("Tool executed: %s", desc)
        return desc, result