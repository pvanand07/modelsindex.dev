"""Bounded public HF-card enrichment. Default is plan-only; --fetch enables requests.

No credentials or paid APIs. Cache states and last-good documents are separate. Retries
are scheduled across invocations, never busy-looped. Every redirect consumes request budget.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .resolver import clean_repo, write_report

MAX_BYTES = 400_000
TTLS = {"ok": 30 * 86400, "not_found": 7 * 86400, "denied": 86400,
        "empty": 86400, "invalid_content": 86400, "too_large": 86400,
        "redirect_error": 86400, "http_error": 86400, "transient_error": 300,
        "rate_limited": 300, "deferred": 0}


def cache_path(data: Path, repo: str) -> Path:
    return data / "cache/identity/hf_cards" / (hashlib.sha256(repo.encode()).hexdigest() + ".json")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def http_get(url: str, timeout: float) -> tuple[int, dict, str]:
    request = Request(url, headers={"User-Agent": "model-identity/1.2 (public model-card research)",
                                    "Accept": "text/plain"})
    try:
        response = build_opener(NoRedirect).open(request, timeout=timeout)
    except HTTPError as error:
        with error:
            return error.code, {k.lower(): v for k, v in error.headers.items()}, ""
    with response:
        content = response.read(MAX_BYTES + 1)
        headers = {k.lower(): v for k, v in response.headers.items()}
        if len(content) > MAX_BYTES:
            return response.status, {**headers, "x-identity-too-large": "true"}, ""
        return response.status, headers, content.decode("utf-8", errors="replace")


def plan(report: dict) -> list[dict]:
    """Fetch only candidate cards whose absence is the remaining constraint failure."""
    groups = report.get("artifact_enrichment_queue")
    if groups is None:
        raise ValueError("Rebuild identity report with resolver >=1.2 before planning enrichment")
    by_repo = {}
    for group in groups:
        if group["action"] != "fetch_cards":
            continue
        for repo in group["repos"]:
            if clean_repo("https://huggingface.co/" + repo) != repo:
                raise ValueError("Invalid queued repository: " + repo)
            item = by_repo.setdefault(repo, {"repo": repo, "digests": set(), "decision_keys": set()})
            item["digests"].update(group["digests"])
            item["decision_keys"].add(group["decision_key"])
    return [{"repo": repo, "digests": sorted(v["digests"]), "decision_keys": sorted(v["decision_keys"])}
            for repo, v in sorted(by_repo.items(), key=lambda pair: (-len(pair[1]["digests"]), pair[0]))]


def retry_delay(state: str, headers: dict, now: float) -> float:
    if state != "rate_limited":
        return TTLS[state]
    value = headers.get("retry-after", "")
    try:
        return max(60, float(value))
    except ValueError:
        from email.utils import parsedate_to_datetime
        try:
            return max(60, parsedate_to_datetime(value).timestamp() - now)
        except (TypeError, ValueError, OverflowError):
            return TTLS[state]


def run(report: dict, data: Path, *, fetch=False, max_repos=10, max_requests=20,
        max_seconds=60, timeout=10, retry_transient=False, transport=http_get, clock=time.time) -> dict:
    if min(max_repos, max_requests, max_seconds, timeout) <= 0:
        raise ValueError("Budgets and timeout must be positive")
    started = clock()
    requests = 0
    items = []
    eligible = plan(report)
    throttle_path = data / "cache/identity/rate_limit.json"
    throttle = json.loads(throttle_path.read_bytes()) if throttle_path.exists() else {}
    if throttle.get("retry_at", 0) > clock():
        return {"schema_version": 1, "fetch_enabled": fetch, "identity_snapshot": report.get("evidence_snapshot_sha256"),
                "summary": {"eligible_repos": len(eligible), "requests": 0, "unprocessed_repos": 0,
                            "actions": {"rate_limit_deferred": len(eligible)}},
                "items": [{**candidate, "action": "rate_limit_deferred", "retry_at": throttle["retry_at"]}
                          for candidate in eligible]}
    attempted_repos = 0
    for candidate in eligible:
        repo = candidate["repo"]
        path = cache_path(data, repo)
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if previous and previous.get("repo") != repo:
            raise ValueError("Cache repository mismatch: " + str(path))
        fresh = previous.get("retry_at", 0) > clock()
        if retry_transient and previous.get("status") == "transient_error":
            fresh = False
        item = {**candidate, "prior_state": previous.get("status"), "action": "cached" if fresh else "fetch"}
        items.append(item)
        if fresh:
            continue
        if attempted_repos >= max_repos or requests >= max_requests or clock() - started >= max_seconds:
            item["action"] = "budget_deferred"
            continue
        attempted_repos += 1
        if not fetch:
            item["action"] = "would_fetch"
            continue
        entry = {**previous, "schema_version": 1, "repo": repo}
        attempts = list(previous.get("attempts") or [])
        pending = previous.get("pending") or {}
        branch = pending.get("branch", "main")
        url = pending.get("url") or f"https://huggingface.co/{repo}/raw/{branch}/README.md"
        redirects = 0
        state, headers = "deferred", {}
        while requests < max_requests and clock() - started < max_seconds:
            # Pending cache URLs and redirects must stay on public HF HTTPS endpoints.
            parsed = urlsplit(url)
            if parsed.scheme != "https" or parsed.hostname != "huggingface.co" or parsed.username or parsed.password or parsed.port not in (None, 443):
                state = "redirect_error"
                break
            requests += 1
            now = clock()
            try:
                code, headers, text = transport(url, min(timeout, max_seconds - (now - started)))
                error_type = None
            except (URLError, TimeoutError, ConnectionError, OSError) as error:
                code, headers, text, error_type = None, {}, "", type(error).__name__
            attempts.append({"at": now, "url": url, "branch": branch, "http_status": code, "error_type": error_type})
            if code in (301, 302, 303, 307, 308):
                if not headers.get("location") or redirects >= 3:
                    state = "redirect_error"
                    break
                redirects += 1
                url = urljoin(url, headers["location"])
                continue
            if code == 404 and branch == "main":
                branch = "master"
                url = f"https://huggingface.co/{repo}/raw/master/README.md"
                continue
            if code is None or code == 408 or code is not None and code >= 500:
                state = "transient_error"
            elif code == 429:
                state = "rate_limited"
            elif code in (401, 403):
                state = "denied"
            elif code == 404:
                state = "not_found"
            elif code != 200:
                state = "http_error"
            elif headers.get("x-identity-too-large"):
                state = "too_large"
            elif not text.strip():
                state = "empty"
            elif "text/html" in headers.get("content-type", "").lower() or text.lstrip().lower().startswith(("<!doctype html", "<html")):
                state = "invalid_content"
            else:
                state = "ok"
                entry["document"] = {"text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "fetched_at": now, "branch": branch, "revision": headers.get("x-repo-commit"), "url": url}
            break
        finished = clock()
        entry.update(status=state, last_attempt_at=attempts[-1]["at"] if attempts else None,
                     retry_at=finished + retry_delay(state, headers, finished), attempts=attempts,
                     pending={"branch": branch, "url": url} if state == "deferred" else None)
        write_report(path, entry)
        item.update(action="attempted", state=state, has_document=bool(entry.get("document")))
        # Honor server backoff globally, not just for the single repo that hit the limit.
        if state == "rate_limited":
            write_report(throttle_path, {"retry_at": entry["retry_at"], "repo": repo, "at": finished})
            break
    return {"schema_version": 1, "fetch_enabled": fetch, "identity_snapshot": report.get("evidence_snapshot_sha256"),
            "summary": {"eligible_repos": len(eligible), "requests": requests,
                        "unprocessed_repos": len(eligible) - len(items),
                        "actions": dict(Counter(i["action"] for i in items))}, "items": items}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--retry-transient", action="store_true", help="Retry transient transport failures after fixing connectivity; does not bypass rate limits")
    parser.add_argument("--max-repos", type=int, default=10)
    parser.add_argument("--max-requests", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=60)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args(argv)
    report = json.loads(args.identity.read_bytes())
    protected = {args.identity.resolve()} | {(args.data_dir / p).resolve() for p in report.get("input_files", {})}
    if args.out.resolve() in protected or cache_path(args.data_dir, "org/repo").parent.resolve() in args.out.resolve().parents:
        parser.error("Output must not overwrite evidence or fetch cache files")
    result = run(report, args.data_dir, fetch=args.fetch, max_repos=args.max_repos, max_requests=args.max_requests,
                 max_seconds=args.max_seconds, timeout=args.timeout, retry_transient=args.retry_transient)
    write_report(args.out, result)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
