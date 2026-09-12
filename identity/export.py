"""Create a production-catalog preview with independent identity evidence attached.

The legacy links remain untouched. Default export includes upstream repos only when the
artifact decision is semantically verified. --include-provisional is explicit and labeled.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .resolver import write_report


def upstream_for(identity: dict, artifact: dict, include_provisional: bool) -> dict | None:
    chosen = artifact["upstream"]
    if chosen.get("status") != "likely" or not chosen.get("repo"):
        return None
    support = []
    for family, reference in artifact.get("family_decisions", {}).items():
        detail = identity["artifact_decisions"].get(reference["decision_key"])
        if not detail or detail.get("repo") != chosen["repo"]:
            continue
        method = detail.get("method")
        if method == "semantic_verified" or include_provisional:
            support.append({"family": family, "decision_key": reference["decision_key"],
                            "method": method, "semantic": detail.get("semantic")})
    if not support:
        return None
    verified = all(s["method"] == "semantic_verified" for s in support)
    return {"repo": chosen["repo"], "url": "https://huggingface.co/" + chosen["repo"],
            "confidence": "semantic_verified" if verified else "provisional",
            "support": support, "supporting_releases": chosen.get("supporting_releases", [])}


def export(identity: dict, legacy: dict, include_provisional=False) -> dict:
    rows, counts = [], Counter()
    seen = set()
    for model in legacy["models"]:
        digest = model["digest"]
        if digest in seen:
            raise ValueError("Duplicate production digest: " + digest)
        seen.add(digest)
        artifact = identity.get("by_digest", {}).get(digest)
        evidence = None
        if artifact:
            upstream = upstream_for(identity, artifact, include_provisional)
            evidence = {"schema_version": 1, "upstream": upstream,
                        "exact_file_uploads": artifact.get("file_matches", []),
                        "status": artifact["upstream"]["status"],
                        "identity_report_snapshot": identity.get("evidence_snapshot_sha256")}
            counts["rows_with_exact_uploads"] += bool(evidence["exact_file_uploads"])
            counts["rows_with_exported_upstream"] += bool(upstream)
            if upstream:
                counts[upstream["confidence"]] += 1
        row = dict(model)
        row["identity_v2"] = evidence
        rows.append(row)
    counts["rows"] = len(rows)
    counts["identity_report_only_artifacts"] = len(set(identity.get("by_digest", {})) - seen)
    return {"schema_version": legacy.get("schema_version"), "identity_export_schema": 1,
            "identity_resolver_version": identity.get("resolver_version"),
            "identity_evidence_snapshot": identity.get("evidence_snapshot_sha256"),
            "include_provisional": include_provisional, "summary": dict(counts), "models": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-provisional", action="store_true")
    args = parser.parse_args(argv)
    if args.out.resolve() in (args.identity.resolve(), args.models.resolve()):
        parser.error("Output must differ from both inputs")
    identity_bytes, model_bytes = args.identity.read_bytes(), args.models.read_bytes()
    result = export(json.loads(identity_bytes), json.loads(model_bytes), args.include_provisional)
    result["input_hashes"] = {"identity": hashlib.sha256(identity_bytes).hexdigest(),
                              "models": hashlib.sha256(model_bytes).hexdigest()}
    write_report(args.out, result)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
