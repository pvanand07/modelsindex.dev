#!/usr/bin/env python3
"""Rebuild model identity from fetched evidence; no imports from legacy resolvers.

python -m identity resolver --data-dir data --out data/out/identity_v2.json
All paths derive from --data-dir. This module makes no network or LLM calls.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

VERSION = "1.3.0"
QUANT = re.compile(r"(?:^|[-_:])(?:f16|fp16|bf16|fp32|f32|fp8|q\d(?:_[a-z0-9]+){0,2}|iq\d(?:_[a-z0-9]+){0,2}|mxfp\d|nvfp\d)$", re.I)
SIZE = re.compile(r"(?<![a-z0-9.])\d+(?:\.\d+)?(?:x\d+(?:\.\d+)?)?[bm](?![a-z0-9])", re.I)
URL = re.compile(r"https?://[^\s<>\]\)\"']+", re.I)
BARE = re.compile(r"(?<![\w/.-])([\w.-]+/[\w.-]+)(?![\w/.-])")
SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
RESERVED = set("datasets spaces papers blog learn collections docs tasks chat join pricing enterprise settings posts welcome inference-endpoints inference-api api organizations".split())
MODES = {"base", "instruct", "chat", "thinking"}
GENERIC = {"latest", "default"}
VERSION_TOKEN = re.compile(r"v\d+(?:\.\d+)*\Z", re.I)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def release_key(family: str, tag: str) -> str:
    # A quant-only alias is deliberately not treated as a specific checkpoint.
    return family + ":" + (QUANT.sub("", tag) or "default")


def clean_repo(url: str) -> str | None:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or parsed.hostname != "huggingface.co":
        return None
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2 or parts[0].lower() in RESERVED:
        return None
    if not all(SEGMENT.fullmatch(p) for p in parts[:2]):
        return None
    return "/".join(parts[:2])


def features(text: str) -> dict:
    sizes = {m.group().lower() for m in SIZE.finditer(text)}
    rest = SIZE.sub(" ", text.lower())
    words = set(re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", rest))
    return {"sizes": sorted(sizes), "modes": sorted(words & MODES),
            "qualifiers": sorted(words - GENERIC - MODES)}


def inventory(rows: list[dict]) -> tuple[dict, dict]:
    artifacts, releases = {}, {}
    for row in rows:
        digest = row.get("weight_digest") or row.get("digest")
        if not digest or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            continue
        refs = list(row.get("aliases") or [])
        if row.get("model") and row.get("tag"):
            refs.append(f"{row['model']}:{row['tag']}")
        if row.get("ref"):
            refs.append(row["ref"])
        artifact = artifacts.setdefault(digest, {"refs": set(), "releases": set(), "names": set(), "architectures": set(),
                                                 "expert_counts": set(), "parameter_counts": set()})
        if row.get("expert_count"):
            artifact["expert_counts"].add(row["expert_count"])
        if row.get("param_count"):
            artifact["parameter_counts"].add(row["param_count"])
        if row.get("name"):
            artifact["names"].add(row["name"])
        if row.get("arch"):
            artifact["architectures"].add(row["arch"])
        for ref in refs:
            if ":" not in ref:
                continue
            family, tag = ref.split(":", 1)
            key = release_key(family, tag)
            artifact["refs"].add(ref)
            artifact["releases"].add(key)
            release = releases.setdefault(key, {"family": family, "tag": key.split(":", 1)[1], "digests": set(), "refs": set()})
            release["digests"].add(digest)
            release["refs"].add(ref)
    return artifacts, releases


def extract_candidates(text: str, family: str, source: str, kind: str) -> list[dict]:
    """Keep the origin and a context window; a mention alone is not an identity proof."""
    found = []
    spans = [(m.start(), m.end(), clean_repo(m.group())) for m in URL.finditer(text)]
    # Bare tokens are a lower-confidence discovery source and must resemble the family.
    spans += [(m.start(), m.end(), clean_repo("https://huggingface.co/" + m.group(1)))
              for m in BARE.finditer(text) if norm(family) in norm(m.group(1).split("/")[-1])]
    seen = set()
    for start, end, repo in spans:
        if not repo or repo in seen:
            continue
        if repo.lower().endswith((".md", ".py", ".json", ".png", ".svg", ".yaml", ".sh")):
            continue
        seen.add(repo)
        found.append({"repo": repo, "source": source, "kind": kind,
                      "excerpt": text[max(0, start - 160):min(len(text), end + 160)]})
    return found


def alias_constraints(tags: list[str], expert_counts=()) -> dict:
    """Combine only same-artifact alias facts; normalize ambiguous notation with evidence."""
    parsed = [features(tag) for tag in tags]
    context_tokens = {s for f in parsed for s in f["sizes"] if s.endswith("m")
                      and any(v.endswith("b") for v in f["sizes"])}
    notes = []
    for f in parsed:
        for token in list(f["sizes"]):
            if token in context_tokens:
                f["sizes"].remove(token)
                f["qualifiers"].append(token)
                notes.append("context_token_from_explicit_alias:" + token)
            moe = re.fullmatch(r"(\d+)x(\d+(?:\.\d+)?b)", token)
            if moe:
                experts, size = moe.groups()
                if int(experts) in expert_counts and any(size in p["sizes"] and experts + "e" in p["qualifiers"] for p in parsed):
                    f["sizes"].remove(token)
                    f["sizes"].append(size)
                    f["qualifiers"].append(experts + "e")
                    notes.append("moe_equivalence_from_alias_and_header:" + token)
    combined = {k: sorted({v for f in parsed for v in f[k]}) for k in ("sizes", "modes", "qualifiers")}
    return {"features": combined, "conflict": len(combined["sizes"]) > 1 or len(combined["modes"]) > 1,
            "notes": sorted(set(notes)), "tags": sorted(set(tags))}


def evaluate(family: str, tag: str, repo: str, card: str | None, target: dict | None = None) -> list[str]:
    """Conservative contradictions and missing-evidence gates, not a similarity score."""
    target = target if target is not None else features(tag)
    candidate = features(repo.split("/", 1)[1])
    # An explicit context token in the alias constraints is not a parameter count.
    for token in list(candidate["sizes"]):
        if token.endswith("m") and token in target["qualifiers"]:
            candidate["sizes"].remove(token)
            candidate["qualifiers"].append(token)
    reasons = []
    if tag in GENERIC and not any(target.values()):
        reasons.append("unspecified_release")
    if norm(family) not in norm(repo.split("/", 1)[1]):
        reasons.append("family_name_not_confirmed")
    if set(candidate["qualifiers"]) & {"gguf", "gptq", "awq", "exl2"}:
        reasons.append("converted_repository_not_canonical")
    if target["sizes"] and set(target["sizes"]) != set(candidate["sizes"]):
        reasons.append("size_mismatch_or_missing")
    if target["modes"] and set(target["modes"]) != set(candidate["modes"]):
        reasons.append("variant_mismatch_or_missing")
    if not target["modes"] and candidate["modes"]:
        reasons.append("variant_unspecified")
    target_versions = {q for q in target["qualifiers"] if VERSION_TOKEN.fullmatch(q)}
    candidate_versions = {q for q in candidate["qualifiers"] if VERSION_TOKEN.fullmatch(q)}
    if candidate_versions and not target_versions:
        reasons.append("version_unspecified")
    if not set(target["qualifiers"]).issubset(candidate["qualifiers"]):
        reasons.append("release_qualifier_missing")
    if not card or not card.strip():
        reasons.append("model_card_unavailable")
    return reasons


def resolve(family: str, tag: str, candidates: list[dict], cards: dict[str, str], target: dict | None = None) -> dict:
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["repo"]].append(candidate)
    target = target if target is not None else features(tag)
    checks = {repo: evaluate(family, tag, repo, cards.get(repo), target) for repo in sorted(grouped)}
    # A bare size does not mean "base". If candidates expose named variants, an
    # unspecified tag cannot choose the base repo merely because variants were rejected.
    if not target["modes"] and any(features(r.split("/", 1)[1])["modes"] and
                                   set(checks[r]) <= {"variant_unspecified", "model_card_unavailable"} for r in grouped):
        for reasons in checks.values():
            if "variant_unspecified" not in reasons:
                reasons.append("variant_unspecified")
    accepted = [repo for repo, reasons in checks.items() if not reasons]
    pending_cards = [repo for repo, reasons in checks.items() if reasons == ["model_card_unavailable"]]
    # Missing evidence is not a rejection. Do not select a repo simply because an
    # equally name-compatible candidate's card was unavailable or access-gated.
    status = "ambiguous" if len(accepted) + len(pending_cards) > 1 else "likely" if len(accepted) == 1 else "unresolved"
    # Include evidence and card hashes so changing input invalidates the decision naturally.
    key = fingerprint({"version": VERSION, "family": family, "tag": tag, "target": target,
                       "candidates": candidates, "cards": {r: fingerprint(cards.get(r)) for r in checks}})
    return {"status": status, "repo": accepted[0] if status == "likely" else None,
            "method": "release_name_constraints" if status == "likely" else None,
            "decision_key": key, "checks": checks, "evidence": dict(sorted(grouped.items()))}


def load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def build(data: Path) -> dict:
    raw_path = data / "out/models.jsonl"
    raw_bytes = raw_path.read_bytes()
    input_files = {"out/models.jsonl": hashlib.sha256(raw_bytes).hexdigest()}

    def read(path: Path, default):
        content = path.read_bytes() if path.exists() else None
        input_files[path.relative_to(data).as_posix()] = hashlib.sha256(content).hexdigest() if content is not None else None
        return json.loads(content) if content is not None else default

    rows = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
    artifacts, releases = inventory(rows)
    families = sorted({r["family"] for r in releases.values()})
    library = read(data / "out/library.json", {}).get("families", {})
    search = read(data / "identity/evidence/search_audits/brave_search_results.json", {})
    cards, candidates = {}, {f: [] for f in families}
    for path in sorted((data / "cache/identity/imported/cards").glob("*.json")):
        text = read(path, {}).get("text")
        if text:
            cards[path.stem.replace("__", "/", 1)] = text
    for family in families:
        candidates[family] += extract_candidates((library.get(family) or {}).get("readme") or "", family,
                                                 f"https://ollama.com/library/{family}", "ollama_readme")
        content = read(data / f"identity/evidence/link_content/{family}.json", {})
        for kind in ("github", "homepage"):
            entry = content.get(kind) or {}
            candidates[family] += extract_candidates(entry.get("content") or "", family, entry.get("url") or kind, kind)
        for repo, entry in (content.get("hf") or {}).items():
            if not isinstance(entry, dict):
                continue
            if entry.get("content"):
                cards[repo] = entry["content"]
            candidates[family].append({"repo": repo, "source": entry.get("url") or f"https://huggingface.co/{repo}",
                                       "kind": "saved_hf_card", "excerpt": ""})
            candidates[family] += extract_candidates(entry.get("content") or "", family, f"https://huggingface.co/{repo}", "hf_card_link")
        for result in (search.get(family) or {}).get("results", []):
            repo = clean_repo(result.get("url") or "")
            if repo:
                candidates[family].append({"repo": repo, "source": result["url"], "kind": "cached_search",
                                           "excerpt": (result.get("description") or result.get("title") or "")[:320]})
    for path in sorted((data / "cache/identity/search").glob("*.json")):
        entry = read(path, {})
        family = entry.get("family")
        if family not in candidates or entry.get("status") != "ok":
            continue
        for result in entry.get("results", []):
            repo = clean_repo(result.get("url") or "")
            if repo:
                candidates[family].append({"repo": repo, "source": result["url"], "kind": "identity_search",
                                           "excerpt": (result.get("description") or result.get("title") or "")[:320]})

    # Independently fetched cards override older legacy content only when integrity matches.
    for path in sorted((data / "cache/identity/hf_cards").glob("*.json")):
        entry = read(path, {})
        repo, document = entry.get("repo"), entry.get("document") or {}
        text = document.get("text")
        if not text:
            continue
        if not repo or clean_repo("https://huggingface.co/" + repo) != repo:
            raise ValueError(f"Invalid enriched card repository in {path}")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != document.get("sha256"):
            raise ValueError(f"Enriched card checksum mismatch in {path}")
        cards[repo] = text

    semantic_cache = {}
    for path in sorted((data / "cache/identity/semantic").glob("*.json")):
        entry = read(path, {})
        if entry.get("decision_key"):
            semantic_cache[entry["decision_key"]] = entry

    exact = {}
    for digest, artifact in sorted(artifacts.items()):
        response = read(data / "cache/identity/imported/hash_lookup" / (digest.split(":")[1] + ".json"), {}) or {}
        if response.get("hex", digest.split(":")[1]) != digest.split(":")[1]:
            raise ValueError(f"Reverse hash cache identity mismatch for {digest}")
        matches = []
        for match in response.get("matches", []):
            repo = clean_repo(f"https://huggingface.co/{match.get('org', '')}/{match.get('name', '')}")
            if match.get("source") != "hf" or not repo:
                continue
            hit = {"repo": repo, "commit_sha": match.get("commit_sha"), "path": match.get("path"),
                   "confidence": "byte_verified", "source": "cached_reverse_hash"}
            if hit not in matches:
                matches.append(hit)
            for family in {releases[r]["family"] for r in artifact["releases"]}:
                candidates[family].append({"repo": repo, "source": digest, "kind": "hash_candidate", "excerpt": ""})
        exact[digest] = sorted(matches, key=lambda h: (h["repo"], h["commit_sha"] or "", h["path"] or ""))

    # Stable deduplication makes decisions independent of input ordering and duplicate mentions.
    for family in families:
        candidates[family] = [json.loads(s) for s in sorted({json.dumps(c, sort_keys=True) for c in candidates[family]})]
    decisions = {}
    for key, release in sorted(releases.items()):
        decision = resolve(release["family"], release["tag"], candidates[release["family"]], cards)
        decisions[key] = {**decision, "digests": sorted(release["digests"]), "refs": sorted(release["refs"])}

    by_digest, constraint_cache = {}, {}
    for digest, artifact in sorted(artifacts.items()):
        family_decisions = {}
        for family in sorted({releases[r]["family"] for r in artifact["releases"]}):
            tags = [releases[r]["tag"] for r in artifact["releases"] if releases[r]["family"] == family]
            constraints = alias_constraints(tags, artifact["expert_counts"])
            cache_key = fingerprint({"family": family, "constraints": constraints})
            if cache_key not in constraint_cache:
                result = resolve(family, "default", candidates[family], cards, constraints["features"])
                if constraints["conflict"]:
                    result.update(status="conflict", repo=None, method=None)
                result["constraints"] = constraints
                result["decision_key"] = fingerprint({"resolution": result["decision_key"], "constraints": constraints})
                semantic = semantic_cache.get(result["decision_key"])
                if semantic and semantic.get("repo") == result.get("repo"):
                    verdict = semantic.get("verdict") or {}
                    reason = str(verdict.get("reason") or "").strip()
                    quote = str(verdict.get("quote") or "").strip()
                    if (verdict.get("exact_release") == "yes" and verdict.get("upstream") == "yes"
                            and quote and len(reason) >= 20 and reason != "..."):
                        result["semantic"] = semantic
                        result["method"] = "semantic_verified"
                    elif verdict.get("exact_release") == "no" or verdict.get("upstream") == "no":
                        result["semantic"] = semantic
                        result.update(status="rejected", repo=None, method=None)
                constraint_cache[cache_key] = result
            family_decisions[family] = constraint_cache[cache_key]
        linked = {d["repo"] for d in family_decisions.values() if d["repo"]}
        identity_conflict = any(d["status"] == "conflict" for d in family_decisions.values())
        # Only same-digest aliases share decisions. Never promote another quant to byte_verified.
        status = "conflict" if identity_conflict or len(linked) > 1 else "likely" if linked else "unresolved"
        by_digest[digest] = {**{k: sorted(v) for k, v in artifact.items()}, "file_matches": exact[digest],
                             "family_decisions": {f: {"status": d["status"], "repo": d["repo"],
                                                        "decision_key": d["decision_key"]} for f, d in family_decisions.items()},
                             "upstream": {"status": status, "repo": next(iter(linked)) if status == "likely" else None,
                                          "alias_identity_conflict": identity_conflict,
                                          "supporting_releases": sorted(r for r in artifact["releases"]
                                                                         if family_decisions[releases[r]["family"]]["repo"]),
                                          "conflicting_repos": sorted(linked) if len(linked) > 1 else []}}
    queue = []
    for key, decision in decisions.items():
        if decision["status"] == "likely":
            continue
        missing = sorted(r for r, reasons in decision["checks"].items() if reasons == ["model_card_unavailable"])
        queue.append({"release": key, "reason": decision["status"], "action": "fetch_cards" if missing else "review_or_search",
                      "repos": missing, "decision_key": decision["decision_key"]})
    # Network work is driven by unresolved artifacts, not generic releases already covered
    # by explicit aliases. Merge identical constraint decisions across quantizations.
    artifact_queue = {}
    artifact_decisions = {d["decision_key"]: d for d in constraint_cache.values()}
    for digest, artifact in by_digest.items():
        if artifact["upstream"]["status"] == "likely":
            continue
        for family, reference in artifact["family_decisions"].items():
            key = reference["decision_key"]
            decision = artifact_decisions[key]
            conflict = artifact["upstream"]["status"] == "conflict"
            missing = sorted(r for r, reasons in decision["checks"].items() if reasons == ["model_card_unavailable"])
            group_key = key + (":conflict" if conflict else "")
            entry = artifact_queue.setdefault(group_key, {"family": family, "decision_key": key,
                "action": "review_conflict" if conflict else "fetch_cards" if missing else
                          "semantic_verify" if decision["status"] == "likely" and not decision.get("semantic") else "review_or_search",
                "repos": [] if conflict else missing, "digests": [], "reason": "conflict" if conflict else decision["status"]})
            entry["digests"].append(digest)
    return {"schema_version": 2, "resolver_version": VERSION, "input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "input_files": dict(sorted(input_files.items())), "evidence_snapshot_sha256": fingerprint(input_files),
            "artifact_decisions": artifact_decisions,
            "summary": {"raw_rows": len(rows), "artifacts": len(artifacts), "releases": len(releases),
                        "byte_matched_artifacts": sum(bool(v) for v in exact.values()),
                        "release_statuses": dict(Counter(d["status"] for d in decisions.values())),
                        "artifact_upstream_statuses": dict(Counter(d["upstream"]["status"] for d in by_digest.values())),
                        "queued_releases": len(queue), "queued_artifact_groups": len(artifact_queue)},
            "by_digest": by_digest, "by_release": decisions, "enrichment_queue": queue,
            "artifact_enrichment_queue": list(artifact_queue.values())}


def write_report(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        json.dump(result, handle, ensure_ascii=True, sort_keys=True, indent=2)
        handle.write("\n")
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(args.data_dir)
    if args.out.resolve() in {(args.data_dir / p).resolve() for p in result["input_files"]}:
        parser.error("Output must not overwrite an evidence input")
    write_report(args.out, result)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
