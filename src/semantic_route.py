"""Fork: semantic fallback for turns the lexical router cannot place.

The regex classifiers in action_intents/turn_contract are fast and exact when
they match, but miss paraphrases ("Check the price of the framework desktop",
"did Linda reply to me yet"). This module asks a local Ollama model ONE
multiple-choice question and reads the probability of each option letter from
the first token's logprobs (the Jev/SemIf "System One" pattern), so no text is
generated or parsed. It only runs when the lexical router found nothing, and
any failure returns None so the caller keeps its existing behaviour.

Opt-in via environment (default off):
  ODYSSEUS_SEMANTIC_ROUTER=1
  ODYSSEUS_SEMANTIC_ROUTER_MODEL=qwen3:30b-a3b   (ChatML/Qwen models only)
  ODYSSEUS_SEMANTIC_ROUTER_URL                   (default: OLLAMA_BASE_URL sans /v1)
  ODYSSEUS_SEMANTIC_ROUTER_THRESHOLD=0.9         (min calibrated probability)
  ODYSSEUS_SEMANTIC_ROUTER_TEMPERATURE=1.5       (fit with scripts/eval_semantic_route.py)
  ODYSSEUS_SEMANTIC_ROUTER_TIMEOUT=4
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteOption:
    letter: str
    label: str
    # ToolIntent category for chat→agent escalation ("" = stay as is).
    category: str
    # Turn-contract families offered and granted for this turn.
    families: frozenset[str]
    description: str


# Shell/files are deliberately absent: host access stays with the
# deterministic router. "other" absorbs those so they are not forced into a
# wrong family.
ROUTE_OPTIONS: tuple[RouteOption, ...] = (
    RouteOption("A", "chat", "", frozenset(),
                "conversation, opinions, advice, explanations or general knowledge you can answer yourself"),
    RouteOption("B", "web", "web", frozenset({"search_browser"}),
                "needs a live web search or web page: prices, news, sports results, weather, "
                "anything current or that may have changed recently"),
    RouteOption("C", "memory", "memory", frozenset({"memory"}),
                "tells you a fact about the user to remember for future chats, or asks you to update or "
                "forget one (NOT questions about past conversations)"),
    RouteOption("D", "email", "email", frozenset({"email"}),
                "the user's email: checking for new mail, replies, receipts or confirmations someone sent, "
                "reading, searching, writing, replying to or sending messages"),
    RouteOption("E", "calendar", "calendar", frozenset({"calendar"}),
                "the user's calendar: events, appointments, schedule, what's on a day"),
    RouteOption("F", "notes", "notes", frozenset({"notes", "tasks"}),
                "the user's notes, lists, to-do list, tasks or reminders: writing something down or "
                "checking what is written there"),
    RouteOption("G", "other", "", frozenset(),
                "anything else that needs the computer or the app itself: files, shell commands, "
                "settings, models, documents"),
)
_BY_LETTER = {o.letter: o for o in ROUTE_OPTIONS}

_SYSTEM_PROMPT = (
    "You route messages for a personal assistant that has tools. Pick the ONE option that "
    "best describes what the user's latest message needs. Reply with the option letter only.\n"
    + "\n".join(f"{o.letter} = {o.description}" for o in ROUTE_OPTIONS)
)

_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RouteDecision:
    option: RouteOption
    probability: float
    probabilities: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.9

    @property
    def confident(self) -> bool:
        return self.probability >= self.threshold

    @property
    def actionable(self) -> bool:
        """Confident and names a tool family (not chat/other)."""
        return self.confident and bool(self.option.families)

    def describe(self) -> str:
        return f"semantic route {self.option.label} p={self.probability:.2f}"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def router_model() -> str | None:
    """Configured router model, or None when the router is disabled."""
    if os.getenv("ODYSSEUS_SEMANTIC_ROUTER", "").strip().lower() not in _TRUE:
        return None
    model = os.getenv("ODYSSEUS_SEMANTIC_ROUTER_MODEL", "").strip()
    if not model:
        return None
    if "qwen" not in model.lower():
        # The raw prompt below is ChatML with an empty think block; other
        # templates would score garbage, so stay off rather than mislead.
        logger.warning("semantic router: %s is not a ChatML/Qwen model; router disabled", model)
        return None
    return model


def _base_url() -> str:
    url = os.getenv("ODYSSEUS_SEMANTIC_ROUTER_URL", "").strip()
    if not url:
        url = os.getenv("OLLAMA_BASE_URL", "").strip() or "http://localhost:11434"
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def build_prompt(message: str, previous: str | None = None) -> str:
    user = message.strip()
    if previous and previous.strip():
        user = (f"(Earlier message from the user, for context only: {previous.strip()[:300]})\n\n"
                f"Latest message: {user}")
    # Pre-filling an empty think block keeps hybrid-thinking Qwen models from
    # reasoning first; Ollama's think:false is not honoured for raw prompts.
    return (f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user[:2000]}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:")


def letter_probabilities(top_logprobs: list[dict], temperature: float = 1.0) -> dict[str, float]:
    """Normalise first-token logprobs over the option letters only.

    Temperature > 1 softens an overconfident model (temperature scaling).
    """
    scores: dict[str, float] = {}
    for cand in top_logprobs or []:
        letter = str(cand.get("token", "")).strip().upper()
        if letter in _BY_LETTER and "logprob" in cand:
            # " A" and "A" are distinct tokens; merge their mass.
            p = math.exp(float(cand["logprob"]))
            scores[letter] = scores.get(letter, 0.0) + p
    if not scores:
        return {}
    t = max(temperature, 1e-3)
    scaled = {k: math.exp(math.log(v) / t) for k, v in scores.items()}
    total = sum(scaled.values())
    return {k: v / total for k, v in sorted(scaled.items(), key=lambda kv: -kv[1])}


def decide(probabilities: dict[str, float], threshold: float) -> RouteDecision | None:
    if not probabilities:
        return None
    letter, p = next(iter(probabilities.items()))
    return RouteDecision(_BY_LETTER[letter], p, dict(probabilities), threshold)


def previous_user_message(history) -> str | None:
    """Most recent stored user message (the current one is not in history yet)."""
    for row in reversed(list(history or ())):
        get = row.get if isinstance(row, dict) else (lambda key, default=None, r=row: getattr(r, key, default))
        if get("role") == "user" and isinstance(get("content"), str):
            return get("content")
    return None


async def score_message(message: str, previous: str | None = None, *, model: str,
                        temperature: float = 1.0, timeout: float = 4.0) -> dict[str, float]:
    body = {
        "model": model,
        "prompt": build_prompt(message, previous),
        "raw": True,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 20,
        "options": {"num_predict": 1, "temperature": 0},
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{_base_url()}/api/generate", json=body)
        resp.raise_for_status()
        data = resp.json()
    first = (data.get("logprobs") or [{}])[0]
    return letter_probabilities(first.get("top_logprobs") or [], temperature)


async def semantic_route(message: str, previous: str | None = None) -> RouteDecision | None:
    """Classify an unplaced turn; None when disabled, empty or on any error."""
    model = router_model()
    if not model or not isinstance(message, str) or not message.strip():
        return None
    try:
        probs = await score_message(
            message, previous, model=model,
            temperature=_env_float("ODYSSEUS_SEMANTIC_ROUTER_TEMPERATURE", 1.5),
            timeout=_env_float("ODYSSEUS_SEMANTIC_ROUTER_TIMEOUT", 4.0),
        )
    except Exception as exc:  # fail open: the lexical result stands
        logger.warning("semantic router failed: %s", exc)
        return None
    decision = decide(probs, _env_float("ODYSSEUS_SEMANTIC_ROUTER_THRESHOLD", 0.9))
    if decision is not None:
        logger.info("%s probs=%s", decision.describe(),
                    {k: round(v, 3) for k, v in decision.probabilities.items()})
    return decision
