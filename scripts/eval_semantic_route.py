#!/usr/bin/env python3
"""Fork: score the semantic router on labelled cases and fit its calibration.

Runs every case in tests/fixtures/semantic_route_cases.json against a live
Ollama model, then:
  * fits the temperature (ODYSSEUS_SEMANTIC_ROUTER_TEMPERATURE) that minimises
    negative log-likelihood of the true label,
  * reports accuracy and expected calibration error before/after, and
  * for each threshold, how many tool turns get routed and how many routes
    are wrong (a wrong route offers the wrong tools, so precision matters
    more than coverage).

Usage (no httpx on the host, so run it in the app image):
  docker run --rm --network host -v ~/odysseus:/src:ro,z -w /src \
    --entrypoint python odysseus-local:latest scripts/eval_semantic_route.py \
    --model qwen3:30b-a3b --url http://localhost:11434
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.action_intents import classify_tool_intent  # noqa: E402
from src.semantic_route import ROUTE_OPTIONS, decide, letter_probabilities, score_message  # noqa: E402

CASES = ROOT / "tests" / "fixtures" / "semantic_route_cases.json"
LETTER = {o.label: o.letter for o in ROUTE_OPTIONS}
TOOL_LABELS = {o.label for o in ROUTE_OPTIONS if o.families}


def rescale(probs: dict[str, float], t: float) -> dict[str, float]:
    # Probabilities are already normalised over letters at T=1, so scaling
    # their logs is equivalent to scaling the logprobs.
    return letter_probabilities([{"token": k, "logprob": math.log(max(v, 1e-12))}
                                 for k, v in probs.items()], t)


def nll(rows, t):
    return -sum(math.log(max(rescale(p, t).get(LETTER[c["label"]], 0.0), 1e-9))
                for c, p in rows) / len(rows)


def ece(rows, t, bins=5):
    buckets = [[] for _ in range(bins)]
    for c, p in rows:
        d = decide(rescale(p, t), 0)
        if d:
            buckets[min(int(d.probability * bins), bins - 1)].append(
                (d.probability, d.option.label == c["label"]))
    n = sum(len(b) for b in buckets)
    return sum(abs(sum(q for q, _ in b) / len(b) - sum(ok for _, ok in b) / len(b)) * len(b)
               for b in buckets if b) / max(n, 1)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--url", help="Ollama base URL (sets ODYSSEUS_SEMANTIC_ROUTER_URL)")
    args = ap.parse_args()
    if args.url:
        import os
        os.environ["ODYSSEUS_SEMANTIC_ROUTER_URL"] = args.url

    cases = json.loads(CASES.read_text())
    rows, times = [], []
    for case in cases:
        start = time.time()
        probs = await score_message(case["text"], model=args.model, timeout=120)
        times.append(time.time() - start)
        rows.append((case, probs))
        d = decide(probs, 0)
        got = d.option.label if d else "?"
        lexical = classify_tool_intent(case["text"])
        mark = "ok " if got == case["label"] else "MISS"
        print(f"{mark} {times[-1]:5.2f}s want={case['label']:8} got={got:8} "
              f"p={d.probability if d else 0:.2f} lexical={lexical.category or '-':9} {case['text']}")

    best_t = min((t / 10 for t in range(5, 61)), key=lambda t: nll(rows, t))
    acc = sum(decide(p, 0) is not None and decide(p, 0).option.label == c["label"] for c, p in rows) / len(rows)
    print(f"\n{len(rows)} cases  accuracy={acc:.2%}  median latency={sorted(times)[len(times) // 2]:.2f}s")
    print(f"T=1.0  NLL={nll(rows, 1.0):.3f}  ECE={ece(rows, 1.0):.3f}")
    print(f"T={best_t:.1f}  NLL={nll(rows, best_t):.3f}  ECE={ece(rows, best_t):.3f}   <- suggested temperature")

    print("\nthreshold  routed/tool-cases  wrong-routes  (at suggested T)")
    tool_cases = sum(c["label"] in TOOL_LABELS for c, _ in rows)
    for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        routed = wrong = 0
        for c, p in rows:
            d = decide(rescale(p, best_t), th)
            if d and d.actionable:
                if d.option.label == c["label"]:
                    routed += 1
                else:
                    wrong += 1
        print(f"  {th:.2f}      {routed:3}/{tool_cases}            {wrong}")


if __name__ == "__main__":
    asyncio.run(main())
