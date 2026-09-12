"""Bounded Brave candidate discovery for unresolved artifact decision groups."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .resolver import clean_repo, write_report
from .verify import load_dotenv


def query_for(family: str, decision: dict) -> str:
    values = decision["constraints"]["features"]
    terms = [family, *values.get("sizes", []), *values.get("modes", []), *values.get("qualifiers", [])]
    return "site:huggingface.co " + " ".join(dict.fromkeys(t for t in terms if t and t not in ("text",)))


def plan(report: dict) -> list[dict]:
    details = report.get("artifact_decisions", {})
    out = []
    for group in report.get("artifact_enrichment_queue", []):
        if group["action"] != "review_or_search":
            continue
        decision = details[group["decision_key"]]
        # Search only when no candidate survives local identity constraints. Ambiguity needs review.
        if decision["status"] != "unresolved":
            continue
        query = query_for(group["family"], decision)
        out.append({"family": group["family"], "decision_key": group["decision_key"], "query": query,
                    "digests": group["digests"]})
    return sorted(out, key=lambda x: (-len(x["digests"]), x["family"], x["query"]))


def call_brave(api_key: str, query: str, timeout: float) -> list[dict]:
    url = "https://api.search.brave.com/res/v1/web/search?" + urlencode({"q": query, "count": 10})
    req = Request(url, headers={"Accept": "application/json", "X-Subscription-Token": api_key})
    with urlopen(req, timeout=timeout) as response:
        return ((json.loads(response.read()).get("web") or {}).get("results") or [])


def run(report: dict, data: Path, *, execute=False, limit=5, timeout=20, caller=call_brave, clock=time.time) -> dict:
    load_dotenv(data.parent / ".env")
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if execute and not api_key:
        raise ValueError("BRAVE_SEARCH_API_KEY is required")
    items = []
    for candidate in plan(report)[:limit]:
        key = hashlib.sha256(candidate["query"].encode()).hexdigest()
        cache = data / "cache/identity/search" / f"{key}.json"
        if cache.exists():
            prior = json.loads(cache.read_bytes())
            if prior.get("query") == candidate["query"] and prior.get("decision_key") == candidate["decision_key"]:
                items.append({**candidate, "action": "cached", "repos": [r["repo"] for r in prior.get("results", [])]})
                continue
        item = {**candidate, "action": "would_search"}
        items.append(item)
        if not execute:
            continue
        try:
            raw = caller(api_key, candidate["query"], timeout)
            results, seen = [], set()
            for result in raw:
                repo = clean_repo(result.get("url") or "")
                if not repo or repo in seen:
                    continue
                seen.add(repo)
                results.append({"repo": repo, "url": f"https://huggingface.co/{repo}",
                                "title": str(result.get("title") or "")[:300],
                                "description": str(result.get("description") or "")[:600]})
            entry = {"schema_version": 1, **candidate, "status": "ok", "searched_at": clock(), "results": results}
            write_report(cache, entry)
            item.update(action="searched", repos=[r["repo"] for r in results])
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError, ValueError, json.JSONDecodeError) as error:
            item.update(action="error", error=type(error).__name__)
    return {"schema_version": 1, "execute": execute, "summary": dict(Counter(x["action"] for x in items)), "items": items}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args(argv)
    result = run(json.loads(args.identity.read_bytes()), args.data_dir, execute=args.execute, limit=args.limit)
    write_report(args.out, result)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
