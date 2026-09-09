#!/usr/bin/env python3
"""
Enumerate the Ollama library and build data/out/models.jsonl without downloading weights.

Per tag: manifest (exact layer sizes) -> config blob (param size, quant, family) ->
GGUF header via Range request (architecture fields, tensor-table byte sums).
Everything is cached under data/cache/ keyed by digest, so re-runs only fetch new tags.
Rows are appended to --out as each tag completes; a crash can `--resume` (default).

    python scripts/crawl.py --models llama3.2,qwen3,gemma3     # smoke crawl
    python scripts/crawl.py                                     # full library (resumes if jsonl exists)
    python scripts/crawl.py --no-resume                         # ignore existing jsonl
    python scripts/crawl.py --limit-tags 3 --workers 2
    python scripts/crawl.py --enrich-push --workers 16         # HEAD push times onto existing jsonl
    python scripts/crawl.py --enrich-copy --workers 16        # library description/README onto jsonl + library.json

Stdlib only.
"""
from __future__ import annotations

import argparse
import html as htmlmod
import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_header import USER_AGENT, header_summary  # noqa: E402
from progress import Bar  # noqa: E402

REGISTRY = "https://registry.ollama.ai/v2/library"
SITE = "https://ollama.com"
ROOT = Path(__file__).resolve().parents[1]
# Modal mounts Volume modelindex-data at /data and sets MODELINDEX_DATA=/data.
DATA = Path(os.environ.get("MODELINDEX_DATA", str(ROOT / "data")))
CACHE = DATA / "cache"
OUT = DATA / "out"

MT_MODEL = "application/vnd.ollama.image.model"
MT_PROJECTOR = "application/vnd.ollama.image.projector"

README_MAX = 100_000
META_DESC_RE = re.compile(
    r'<meta\s+[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']',
    re.I,
)

HEADER_FIELDS = (
    "arch", "name", "size_label", "n_layer", "n_embd", "n_head", "n_head_kv", "kv_heads_sum",
    "key_length", "value_length", "key_length_swa", "value_length_swa",
    "context_length", "vocab_size", "expert_count",
    "expert_used_count", "sliding_window", "shared_kv_layers",
    "n_swa_layers", "n_global_layers", "n_kv_alloc_layers",
    "embedding_length_per_layer_input",
    "param_count", "active_params", "ple_params", "tensor_count", "gpu_tensor_count",
    "expert_bytes", "active_weight_bytes", "ple_bytes", "gpu_weight_bytes",
    "kv_local_bytes_per_token_f16", "kv_global_bytes_per_token_f16",
    "kv_bytes_per_token_f16", "tensor_groups_bytes",
    "header_bytes",
)


# --------------------------------------------------------------------------- http + cache
def _http_open(url: str, method: str, timeout: int, retries: int):
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(
                url, method=method, headers={"User-Agent": "Mozilla/5.0 " + USER_AGENT},
            )
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(1.5 * (i + 1))
    assert last is not None
    raise last


def http_get(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    with _http_open(url, "GET", timeout, retries) as r:
        return r.read()


def http_head(url: str, timeout: int = 30, retries: int = 3):
    with _http_open(url, "HEAD", timeout, retries) as r:
        return r.headers


def iso_from_push_unix(raw) -> str | None:
    """Convert registry `ollama-push-time` (unix seconds) to UTC ISO-8601."""
    try:
        unix = int(raw)
    except (TypeError, ValueError):
        return None
    if unix <= 0:
        return None
    return datetime.fromtimestamp(unix, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cached_json(path: Path, ttl_days: float | None, producer):
    """Return JSON from `path` if fresh, else call producer(), store, return."""
    if path.exists():
        if ttl_days is None or (time.time() - path.stat().st_mtime) < ttl_days * 86400:
            with open(path, encoding="utf-8") as f:
                return json.load(f), True
    data = producer()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return data, False


def load_jsonl_by_ref(path: Path) -> dict[str, dict]:
    """Load complete JSONL rows; skip a truncated last line from a crash."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ref = row.get("ref")
            if ref:
                out[ref] = row
    return out


class JsonlStore:
    """Append-only jsonl with resume. fsync after each row so a crash keeps prior rows."""

    def __init__(self, path: Path, resume: bool, total: int = 0):
        self.path = path
        self.lock = threading.Lock()
        self.progress_path = path.parent / "crawl_progress.json"
        self.total = total
        path.parent.mkdir(parents=True, exist_ok=True)
        self.by_ref = load_jsonl_by_ref(path) if resume and path.exists() else {}
        if not resume and path.exists():
            path.unlink()
            self.by_ref = {}
        self._rewrite()
        print(f"[crawl] resume {len(self.by_ref)} complete rows in {path}", flush=True)

    def set_total(self, total: int) -> None:
        self.total = total
        self._write_progress()

    def has(self, ref: str) -> bool:
        return ref in self.by_ref

    def add(self, row: dict) -> None:
        ref = row.get("ref")
        if not ref:
            return
        payload = json.dumps(row) + "\n"
        with self.lock:
            self.by_ref[ref] = row
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            self._write_progress()

    def _rewrite(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for ref in sorted(self.by_ref):
                f.write(json.dumps(self.by_ref[ref]) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def finalize(self) -> None:
        with self.lock:
            self._rewrite()
            self._write_progress()

    def _write_progress(self) -> None:
        rec = {
            "done": len(self.by_ref),
            "total": self.total,
            "pct": round(100 * len(self.by_ref) / max(1, self.total), 1),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "out": str(self.path),
        }
        tmp = self.progress_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec), encoding="utf-8")
        os.replace(tmp, self.progress_path)


class Counters:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.d = {k: 0 for k in (
            "manifest_hits", "manifest_fetches", "header_hits", "header_fetches", "header_failures",
        )}

    def add(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.d[key] += n


def list_tags_cached(model: str, ttl_days: float) -> list[str]:
    tags, _ = cached_json(CACHE / "tags" / f"{model}.json", ttl_days, lambda: list_tags(model))
    return list(tags) if isinstance(tags, list) else []


# --------------------------------------------------------------------------- enumeration
def list_models() -> list[str]:
    html = http_get(f"{SITE}/library").decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="/library/([a-z0-9][a-z0-9._-]*)"', html)))


def list_tags(model: str) -> list[str]:
    html = http_get(f"{SITE}/library/{model}/tags").decode("utf-8", "replace")
    pat = rf'href="/library/{re.escape(model)}:([A-Za-z0-9._-]+)"'
    return sorted(set(re.findall(pat, html)))


def _textarea(html: str, element_id: str) -> str:
    pat = re.compile(
        rf'<textarea\b[^>]*\bid=["\']{re.escape(element_id)}["\'][^>]*>(.*?)</textarea\s*>',
        re.I | re.S,
    )
    match = pat.search(html)
    if not match:
        return ""
    return htmlmod.unescape(match.group(1)).strip()


def rewrite_site_urls(text: str) -> str:
    text = re.sub(r"(\]\()(/assets/)", rf"\1{SITE}/assets/", text)
    text = re.sub(r'(src=["\'])(/assets/)', rf"\1{SITE}/assets/", text)
    return text


def parse_library_html(html: str) -> dict:
    """Pull the family blurb and README markdown from an Ollama library HTML page."""
    meta = META_DESC_RE.search(html)
    description = htmlmod.unescape(meta.group(1)).strip() if meta else ""
    if not description:
        description = _textarea(html, "summary-textarea")
    if len(description) > 255:
        description = description[:255].rstrip()
    readme = rewrite_site_urls(_textarea(html, "editor"))
    if len(readme) > README_MAX:
        readme = readme[:README_MAX].rstrip()
    return {"description": description, "readme": readme}


def get_library_copy(model: str, ttl_days: float) -> tuple[dict, bool]:
    def produce():
        try:
            html = http_get(f"{SITE}/library/{model}").decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"description": "", "readme": ""}
            raise
        return parse_library_html(html)
    data, hit = cached_json(CACHE / "library" / f"{model}.json", ttl_days, produce)
    if not isinstance(data, dict):
        return {"description": "", "readme": ""}, hit
    return {
        "description": str(data.get("description") or ""),
        "readme": str(data.get("readme") or ""),
    }, hit


def write_library_json(path: Path, families: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "families": families,
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def enrich_library_copy(
    store: JsonlStore,
    ttl_days: float,
    workers: int,
    models: list[str] | None = None,
    library_out: Path | None = None,
) -> int:
    """Fetch family description/README once per library name; stamp jsonl + library.json."""
    want = set(models) if models else None
    names = sorted({
        r["model"] for r in store.by_ref.values()
        if r.get("model") and (want is None or r["model"] in want)
    })
    if not names:
        print("[crawl] no library families to enrich", flush=True)
        return 0

    families: dict[str, dict] = {}
    fetched = 0
    bar = Bar("crawl library copy", len(names))
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(get_library_copy, name, ttl_days): name for name in names}
        for f in as_completed(futs):
            name = futs[f]
            try:
                copy, hit = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"[crawl] library page failed for {name}: {e}", file=sys.stderr, flush=True)
                copy, hit = {"description": "", "readme": ""}, True
            families[name] = copy
            if not hit:
                fetched += 1
            bar.update(item=name)
    bar.finish()

    patched = 0
    for row in store.by_ref.values():
        name = row.get("model")
        if name not in families:
            continue
        desc = families[name].get("description") or ""
        if row.get("description") != desc:
            row["description"] = desc
            patched += 1
        else:
            row["description"] = desc
    store.finalize()
    out = library_out or (store.path.parent / "library.json")
    write_library_json(out, families)
    with_desc = sum(1 for v in families.values() if v.get("description"))
    with_readme = sum(1 for v in families.values() if v.get("readme"))
    print(
        f"[crawl] library copy {len(families)} families "
        f"(description={with_desc}, readme={with_readme}, fetches={fetched}) -> {out}",
        flush=True,
    )
    return patched


# --------------------------------------------------------------------------- per-tag work
def get_manifest(model: str, tag: str, ttl_days: float) -> tuple[dict, bool]:
    def produce():
        return json.loads(http_get(f"{REGISTRY}/{model}/manifests/{tag}"))
    return cached_json(CACHE / "manifests" / model / f"{tag}.json", ttl_days, produce)


def get_push_time(model: str, tag: str, ttl_days: float) -> tuple[str | None, bool]:
    """HEAD the tag manifest. GET does not return `ollama-push-time`."""
    def produce():
        headers = http_head(f"{REGISTRY}/{model}/manifests/{tag}")
        unix = headers.get("ollama-push-time")
        iso = iso_from_push_unix(unix)
        return {"unix": int(unix) if iso else None, "pushed_at": iso}
    data, hit = cached_json(CACHE / "push_times" / model / f"{tag}.json", ttl_days, produce)
    if not isinstance(data, dict):
        return None, hit
    return data.get("pushed_at"), hit


def get_config(model: str, digest: str) -> tuple[dict, bool]:
    def produce():
        return json.loads(http_get(f"{REGISTRY}/{model}/blobs/{digest}"))
    return cached_json(CACHE / "config" / f"{digest.replace(':', '_')}.json", None, produce)


def get_header(model: str, digest: str) -> tuple[dict, bool]:
    def produce():
        return header_summary(f"{REGISTRY}/{model}/blobs/{digest}")
    return cached_json(CACHE / "headers" / f"{digest.replace(':', '_')}.json", None, produce)


def build_row(model: str, tag: str, ttl_days: float, stats: Counters) -> dict:
    row: dict = {"model": model, "tag": tag, "ref": f"{model}:{tag}"}
    try:
        manifest, hit = get_manifest(model, tag, ttl_days)
    except Exception as e:  # noqa: BLE001
        row["error"] = f"manifest: {e}"
        return row
    stats.add("manifest_hits" if hit else "manifest_fetches")
    try:
        pushed_at, _ = get_push_time(model, tag, ttl_days)
        if pushed_at:
            row["pushed_at"] = pushed_at
    except Exception:  # noqa: BLE001
        pass
    if not isinstance(manifest, dict):
        row["error"] = f"manifest is {type(manifest).__name__}, not an object"
        return row

    layers = manifest.get("layers") or []
    if not isinstance(layers, list):
        row["error"] = f"manifest layers is {type(layers).__name__}"
        return row
    weight = next((l for l in layers if isinstance(l, dict) and l.get("mediaType") == MT_MODEL), None)
    proj = next((l for l in layers if isinstance(l, dict) and l.get("mediaType") == MT_PROJECTOR), None)
    if not weight:
        row["error"] = "manifest has no model layer"
        return row
    row["weight_digest"] = weight["digest"]
    row["weight_bytes"] = weight["size"]
    row["projector_digest"] = proj["digest"] if proj else None
    row["projector_bytes"] = proj["size"] if proj else 0
    row["config_digest"] = manifest.get("config", {}).get("digest")

    try:
        cfg, _ = get_config(model, row["config_digest"])
        row["config"] = {k: cfg.get(k) for k in ("model_format", "model_family", "model_families", "model_type", "file_type")}
    except Exception as e:  # noqa: BLE001
        row["config"] = None
        row["config_error"] = str(e)
    return row


def enrich_push_times(
    store: JsonlStore,
    ttl_days: float,
    workers: int,
    models: list[str] | None = None,
) -> int:
    """HEAD missing `pushed_at` onto existing jsonl rows. Does not refetch GGUF headers."""
    want = set(models) if models else None
    pending = [
        r for r in store.by_ref.values()
        if r.get("model") and r.get("tag") and not r.get("pushed_at")
        and (want is None or r["model"] in want)
    ]
    if not pending:
        print("[crawl] push times already present", flush=True)
        return 0

    def one(row: dict) -> tuple[str, str | None]:
        ref = row["ref"]
        try:
            pushed_at, _ = get_push_time(row["model"], row["tag"], ttl_days)
        except Exception as e:  # noqa: BLE001
            print(f"[crawl] push-time failed for {ref}: {e}", file=sys.stderr, flush=True)
            return ref, None
        return ref, pushed_at

    filled = 0
    bar = Bar("crawl push times", len(pending))
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, r) for r in pending]
        for f in as_completed(futs):
            ref, pushed_at = f.result()
            if pushed_at:
                store.by_ref[ref]["pushed_at"] = pushed_at
                filled += 1
            bar.update(item=ref)
    bar.finish()
    store.finalize()
    print(f"[crawl] push times filled {filled}/{len(pending)}", flush=True)
    return filled


def attach_header(row: dict, stats: Counters) -> dict:
    if "weight_digest" not in row:
        row["header_ok"] = False
        return row
    try:
        h, hit = get_header(row["model"], row["weight_digest"])
        stats.add("header_hits" if hit else "header_fetches")
        for k in HEADER_FIELDS:
            row[k] = h.get(k)
        row["header_ok"] = True
    except Exception as e:  # noqa: BLE001
        row["header_ok"] = False
        row["header_error"] = str(e)
        stats.add("header_failures")
    return row


def refresh_hybrid_kv(out: Path, model_names: list[str], workers: int = 4) -> int:
    """Re-parse each unique weight digest and patch header-derived fields onto jsonl rows."""
    store = JsonlStore(out, resume=True)
    want = set(model_names)
    rows = [r for r in store.by_ref.values() if r.get("model") in want and r.get("header_ok")]
    if not rows:
        print(f"[crawl] --refresh-hybrid-kv: no header_ok rows for {sorted(want)}", flush=True)
        return 0
    by_digest: dict[str, dict] = {}
    for r in rows:
        d = r.get("weight_digest")
        if d and d not in by_digest:
            by_digest[d] = r
    print(f"[crawl] refresh-hybrid-kv {len(rows)} rows, {len(by_digest)} unique digests", flush=True)
    stats = Counters()
    summaries: dict[str, dict] = {}
    for r in by_digest.values():
        digest = r.get("weight_digest") or ""
        cache_path = CACHE / "headers" / f"{digest.replace(':', '_')}.json"
        if cache_path.exists():
            cache_path.unlink()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(attach_header, dict(r), stats): d for d, r in by_digest.items()}
        for f in as_completed(futs):
            src = f.result()
            d = futs[f]
            if not src.get("header_ok"):
                print(f"[crawl] header failed {src.get('ref')}: {src.get('header_error')}", flush=True)
                continue
            summaries[d] = src
    patched = 0
    for r in rows:
        src = summaries.get(r.get("weight_digest"))
        if not src:
            continue
        for k in HEADER_FIELDS:
            if k in src:
                r[k] = src[k]
        patched += 1
        store.by_ref[r["ref"]] = r
    store.finalize()
    print(f"[crawl] refresh-hybrid-kv patched {patched} rows "
          f"(fetches={stats.d.get('header_fetches')} hits={stats.d.get('header_hits')})", flush=True)
    return 0 if patched else 1


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="comma-separated subset of library model names")
    ap.add_argument("--limit-tags", type=int, default=0, help="max tags per model (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ttl-days", type=float, default=7, help="manifest cache TTL (tags can be re-pushed)")
    ap.add_argument("--refresh", action="store_true", help="ignore manifest/tag cache")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="skip refs already in --out (default: true)")
    ap.add_argument("--enrich-push", action="store_true",
                    help="HEAD ollama-push-time onto existing jsonl rows; skip header crawl")
    ap.add_argument("--enrich-copy", action="store_true",
                    help="fetch Ollama library description/README onto jsonl + library.json; skip header crawl")
    ap.add_argument("--refresh-hybrid-kv", action="store_true",
                    help="re-fetch one header per architecture for --models (default gemma4) and patch jsonl")
    ap.add_argument("--out", default=str(OUT / "models.jsonl"))
    a = ap.parse_args(argv)

    t0 = time.time()
    stats = Counters()
    ttl = 0 if a.refresh else a.ttl_days
    out = Path(a.out)
    if a.refresh_hybrid_kv:
        if not out.exists():
            raise SystemExit(f"missing {out}; run a crawl before --refresh-hybrid-kv")
        names = [m.strip() for m in (a.models or "gemma4").split(",") if m.strip()]
        return refresh_hybrid_kv(out, names, a.workers)
    if a.enrich_copy:
        if not out.exists():
            raise SystemExit(f"missing {out}; run a crawl before --enrich-copy")
        store = JsonlStore(out, resume=True)
        names = [m.strip() for m in a.models.split(",") if m.strip()] if a.models else None
        enrich_library_copy(store, ttl, a.workers, names)
        print(f"[crawl] wrote {out} ({round(time.time() - t0, 1)}s)", flush=True)
        return 0
    if a.enrich_push:
        if not out.exists():
            raise SystemExit(f"missing {out}; run a crawl before --enrich-push")
        store = JsonlStore(out, resume=True)
        names = [m.strip() for m in a.models.split(",") if m.strip()] if a.models else None
        enrich_push_times(store, ttl, a.workers, names)
        print(f"[crawl] wrote {out} ({round(time.time() - t0, 1)}s)", flush=True)
        return 0
    store = JsonlStore(out, resume=a.resume)

    models = a.models.split(",") if a.models else list_models()
    print(f"[crawl] {len(models)} models", flush=True)

    tags_by_model: dict[str, list[str]] = {}
    tags_bar = Bar("crawl tags", len(models))
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(list_tags_cached, m, ttl): m for m in models}
        for f in as_completed(futs):
            m = futs[f]
            try:
                tags = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"[crawl] tags failed for {m}: {e}", file=sys.stderr, flush=True)
                tags = []
            if a.limit_tags:
                tags = tags[: a.limit_tags]
            tags_by_model[m] = tags
            tags_bar.update(item=m)
    tags_bar.finish()

    work = [(m, t) for m, ts in tags_by_model.items() for t in ts]
    n_tags = len(work)
    remaining = [(m, t) for m, t in work if not store.has(f"{m}:{t}")]
    store.set_total(n_tags)
    print(f"[crawl] {n_tags} tags, {len(store.by_ref)} done, {len(remaining)} remaining", flush=True)

    rows: list[dict] = []
    man_bar = Bar("crawl manifests", n_tags, done=n_tags - len(remaining))
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(build_row, m, t, ttl, stats) for m, t in remaining]
        for f in as_completed(futs):
            row = f.result()
            rows.append(row)
            if "weight_digest" not in row:
                row.setdefault("header_ok", False)
                store.add(row)
            man_bar.update(item=row.get("ref", ""))
    man_bar.finish()
    rows.sort(key=lambda r: (r["model"], r["tag"]))

    pending = [r for r in rows if "weight_digest" in r]
    first_row_for: dict[str, dict] = {}
    siblings: dict[str, list[dict]] = {}
    for r in pending:
        d = r["weight_digest"]
        siblings.setdefault(d, []).append(r)
        if d not in first_row_for:
            first_row_for[d] = r
    print(f"[crawl] {len(first_row_for)} unique weight digests still need headers "
          f"(dedupe ratio {n_tags / max(1, len(first_row_for) or 1):.2f}x)", flush=True)

    hdr_bar = Bar("crawl headers", len(first_row_for))
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(attach_header, r, stats): d for d, r in first_row_for.items()}
        for f in as_completed(futs):
            src = f.result()
            d = src.get("weight_digest")
            for r in siblings.get(d, []):
                if r is not src:
                    for k in HEADER_FIELDS + ("header_ok", "header_error"):
                        if k in src:
                            r[k] = src[k]
                r.setdefault("header_ok", False)
                store.add(r)
            hdr_bar.update(item=src.get("ref", ""))
    hdr_bar.finish()

    store.finalize()
    enrich_push_times(store, ttl, a.workers)
    enrich_library_copy(store, ttl, a.workers)
    all_rows = list(store.by_ref.values())
    ok = sum(1 for r in all_rows if r.get("header_ok"))
    summary = {
        "models": len(models),
        "tags": n_tags,
        "unique_digests": len({r.get("weight_digest") for r in all_rows if r.get("weight_digest")}),
        "rows_header_ok": ok,
        "rows_header_ok_pct": round(100 * ok / max(1, len(all_rows)), 1),
        "seconds": round(time.time() - t0, 1),
        **stats.d,
    }
    with open(out.parent / "crawl_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[crawl] " + json.dumps(summary), flush=True)
    print(f"[crawl] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
