"""Semantically verify likely artifact decisions with a constrained, cached LLM verdict."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .resolver import fingerprint, write_report

PROMPT_VERSION = 2


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def card_for(data: Path, family: str, repo: str) -> str | None:
    encoded = repo.replace("/", "__", 1)
    imported = data / "cache/identity/imported/cards" / f"{encoded}.json"
    text = json.loads(imported.read_bytes()).get("text") if imported.exists() else None
    content = data / "identity/evidence/link_content" / f"{family}.json"
    if content.exists():
        text = ((json.loads(content.read_bytes()).get("hf") or {}).get(repo) or {}).get("content") or text
    enriched_dir = data / "cache/identity/hf_cards"
    for path in enriched_dir.glob("*.json"):
        entry = json.loads(path.read_bytes())
        if entry.get("repo") == repo and (entry.get("document") or {}).get("text"):
            text = entry["document"]["text"]
            break
    return text


def plan(report: dict) -> list[dict]:
    items = []
    for key, decision in report.get("artifact_decisions", {}).items():
        if decision.get("status") == "likely" and decision.get("repo") and not decision.get("semantic"):
            items.append({"decision_key": key, "repo": decision["repo"],
                          "constraints": decision["constraints"], "evidence": decision["evidence"]})
    impacts = Counter()
    families = {}
    for artifact in report.get("by_digest", {}).values():
        for family, reference in artifact.get("family_decisions", {}).items():
            impacts[reference["decision_key"]] += 1
            families.setdefault(reference["decision_key"], set()).add(family)
    for item in items:
        item["artifacts"] = impacts[item["decision_key"]]
        item["family"] = sorted(families.get(item["decision_key"], {""}))[0]
    return sorted(items, key=lambda x: (-x["artifacts"], x["decision_key"]))


def call_llm(base_url: str, api_key: str, model: str, prompt: str, timeout: float) -> str:
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 1400,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}]}).encode()
    request = Request(base_url.rstrip("/") + "/chat/completions", data=body, method="POST",
                      headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        message = json.loads(response.read())["choices"][0]["message"]
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM returned no final content")
        return content


def parse_verdict(reply: str, card: str) -> dict:
    if not isinstance(reply, str):
        raise ValueError("LLM verdict is not text")
    text = reply.strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("LLM did not return JSON")
    value = json.loads(match.group())
    allowed = {"yes", "no", "uncertain"}
    if value.get("exact_release") not in allowed or value.get("upstream") not in allowed:
        raise ValueError("Invalid verdict vocabulary")
    quote = str(value.get("quote") or "").strip()
    if quote and quote not in card:
        pattern = r"\s+".join(re.escape(part) for part in quote.split())
        grounded = re.search(pattern, card, re.I)
        if not grounded:
            raise ValueError("Evidence quote is not present in model card")
        quote = grounded.group(0)
    reason = str(value.get("reason") or "").strip()
    if value["exact_release"] == "yes" and value["upstream"] == "yes":
        if not quote:
            raise ValueError("Affirmative verdict requires a grounded evidence quote")
        if len(reason) < 20 or reason == "...":
            raise ValueError("Affirmative verdict requires a substantive reason")
    return {"exact_release": value["exact_release"], "upstream": value["upstream"],
            "quote": quote or None, "reason": reason[:500]}


def prompt_for(item: dict, card: str) -> str:
    evidence = []
    for rows in item["evidence"].values():
        evidence.extend(r.get("excerpt", "")[:300] for r in rows if r.get("excerpt"))
    return (
        "Judge repository identity using only the supplied data. A related family, base model, "
        "quantization, descendant, benchmark mention, or comparison is not the exact release. "
        "Upstream means an original checkpoint/model repository, not a conversion or mirror. "
        "Use uncertain when the card does not establish a claim. Return one JSON object with "
        'exact_release and upstream set to yes/no/uncertain, quote copied verbatim from the card '
        'or empty, and a short reason.\n\n'
        f"Family hint: {item['family']}\nTarget constraints: {json.dumps(item['constraints'], ensure_ascii=False)}\n"
        f"Candidate repo: {item['repo']}\nSource excerpts: {json.dumps(evidence[:6], ensure_ascii=False)}\n\n"
        f"Candidate model card:\n{card[:7000]}"
    )


def run(report: dict, data: Path, *, execute=False, limit=10, model="~z-ai/glm-flash-latest",
        timeout=50, caller=call_llm, clock=time.time) -> dict:
    load_dotenv(data.parent / ".env")
    api_key, base_url = os.environ.get("OPENROUTER_API_KEY"), os.environ.get("OPENROUTER_BASE_URL")
    if execute and (not api_key or not base_url):
        raise ValueError("OPENROUTER_API_KEY and OPENROUTER_BASE_URL are required")
    items = []
    for candidate in plan(report)[:limit]:
        family = candidate["family"]
        card = card_for(data, family, candidate["repo"])
        item = {**candidate, "family": family, "card_available": bool(card), "action": "missing_card" if not card else "would_verify"}
        items.append(item)
        if not card or not execute:
            continue
        input_key = fingerprint({"prompt_version": PROMPT_VERSION, "model": model,
                                 "decision_key": candidate["decision_key"], "card": hashlib.sha256(card.encode()).hexdigest()})
        cache = data / "cache/identity/semantic" / f"{candidate['decision_key']}.json"
        if cache.exists():
            prior = json.loads(cache.read_bytes())
            if prior.get("input_key") == input_key:
                item.update(action="cached", verdict=prior["verdict"])
                continue
        try:
            reply = caller(base_url, api_key, model, prompt_for(candidate, card), timeout)
            verdict = parse_verdict(reply, card)
            entry = {"schema_version": 1, "decision_key": candidate["decision_key"], "repo": candidate["repo"],
                     "input_key": input_key, "model": model, "prompt_version": PROMPT_VERSION,
                     "verified_at": clock(), "verdict": verdict}
            write_report(cache, entry)
            item.update(action="verified", verdict=verdict)
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            item.update(action="error", error=type(error).__name__)
    return {"schema_version": 1, "execute": execute, "summary": dict(Counter(i["action"] for i in items)), "items": items}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--model", default="~z-ai/glm-flash-latest")
    args = parser.parse_args(argv)
    result = run(json.loads(args.identity.read_bytes()), args.data_dir, execute=args.execute, limit=args.limit, model=args.model)
    write_report(args.out, result)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
