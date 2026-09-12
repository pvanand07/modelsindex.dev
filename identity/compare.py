"""Compare captured identity/legacy report bytes by digest without changing either input."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def compare(identity: dict, legacy: dict) -> dict:
    older = {}
    for row in legacy["models"]:
        if row["digest"] in older:
            raise ValueError("Duplicate legacy digest: " + row["digest"])
        older[row["digest"]] = row
    newer = identity["by_digest"]
    rows = []
    for digest in sorted(set(older) & set(newer)):
        old = older[digest]
        hf = (old.get("links") or {}).get("hf") or {}
        release = hf.get("release") or {}
        # A verified upload is not an upstream-publisher label.
        old_repo = release.get("repo") if release.get("confidence") == "likely" else None
        new_repo = newer[digest]["upstream"].get("repo")
        category = ("agree" if old_repo == new_repo else "different") if old_repo and new_repo else (
            "new_only" if new_repo else "legacy_only" if old_repo else "neither")
        rows.append({"digest": digest, "ref": old["ref"], "category": category,
                     "legacy_likely_release": old_repo, "new_likely_upstream": new_repo,
                     "legacy_family_fallback": (hf.get("family") or {}).get("repo") if not release else None,
                     "legacy_exact_upload": release.get("repo") if release.get("confidence") == "verified" else None,
                     "new_status": newer[digest]["upstream"]["status"]})
    return {"summary": {"shared_artifacts": len(rows), "identity_only_artifacts": len(set(newer) - set(older)),
                        "legacy_only_artifacts": len(set(older) - set(newer)),
                        "upstream_comparison": dict(Counter(r["category"] for r in rows))},
            "identity_only_digests": sorted(set(newer) - set(older)),
            "legacy_only_digests": sorted(set(older) - set(newer)), "rows": rows,
            "interpretation": "Agreement and coverage deltas are not accuracy measurements. Inputs may reflect different fetch dates."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--legacy", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.resolve() in (args.identity.resolve(), args.legacy.resolve()):
        parser.error("Output must differ from both input files")
    identity_bytes, legacy_bytes = args.identity.read_bytes(), args.legacy.read_bytes()
    result = compare(json.loads(identity_bytes), json.loads(legacy_bytes))
    result["input_hashes"] = {"identity": hashlib.sha256(identity_bytes).hexdigest(),
                              "legacy": hashlib.sha256(legacy_bytes).hexdigest()}
    from .resolver import write_report
    write_report(args.out, result)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
