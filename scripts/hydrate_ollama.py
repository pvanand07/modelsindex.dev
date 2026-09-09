#!/usr/bin/env python3
"""Write registry blobs into Ollama's on-disk layout without `ollama pull`.

Downloads config + layers in parallel into $OLLAMA_MODELS/blobs/sha256-..., then
writes manifests/registry.ollama.ai/library/{model}/{tag}. Existing blobs with the
right size are skipped. Stdlib only.

    python scripts/hydrate_ollama.py --models llama3.2:1b,llama3.2:3b
    python scripts/hydrate_ollama.py --models llama3.1:8b --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crawl import REGISTRY, get_manifest  # noqa: E402
from gguf_header import USER_AGENT  # noqa: E402
from progress import Bar, ProgressFile, fmt_bytes, fmt_dur  # noqa: E402

CHUNK = 1024 * 1024
HOST = "registry.ollama.ai"


def models_dir() -> Path:
    return Path(os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama" / "models")))


def split_ref(ref: str) -> tuple[str, str]:
    if ":" not in ref:
        raise ValueError(f"expected model:tag, got {ref!r}")
    model, tag = ref.split(":", 1)
    return model, tag


def blob_path(root: Path, digest: str) -> Path:
    return root / "blobs" / digest.replace(":", "-")


def manifest_path(root: Path, model: str, tag: str) -> Path:
    return root / "manifests" / HOST / "library" / model / tag


def blob_present(path: Path, size: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size == size
    except OSError:
        return False


def collect_blobs(refs: list[str], ttl_days: float = 30.0) -> tuple[list[dict], dict[str, dict]]:
    """Unique blobs plus per-ref manifests. URL uses the repo that listed the blob."""
    blobs: dict[str, dict] = {}
    manifests: dict[str, dict] = {}
    for ref in refs:
        model, tag = split_ref(ref)
        manifest, _ = get_manifest(model, tag, ttl_days)
        if not isinstance(manifest, dict):
            raise RuntimeError(f"{ref}: manifest is {type(manifest).__name__}")
        manifests[ref] = manifest
        items: list[dict] = []
        cfg = manifest.get("config")
        if isinstance(cfg, dict) and cfg.get("digest"):
            items.append(cfg)
        for layer in manifest.get("layers") or []:
            if isinstance(layer, dict) and layer.get("digest"):
                items.append(layer)
        for item in items:
            digest = item["digest"]
            size = int(item.get("size") or 0)
            if digest not in blobs:
                blobs[digest] = {
                    "digest": digest,
                    "size": size,
                    "url": f"{REGISTRY}/{model}/blobs/{digest}",
                }
            elif size and blobs[digest]["size"] and size != blobs[digest]["size"]:
                raise RuntimeError(f"{digest}: size mismatch {blobs[digest]['size']} vs {size}")
            elif size and not blobs[digest]["size"]:
                blobs[digest]["size"] = size
    return list(blobs.values()), manifests


def download_blob(url: str, dest: Path, expected: int, on_bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last: Exception | None = None
    for i in range(3):
        got = 0
        try:
            if tmp.exists():
                tmp.unlink()
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 " + USER_AGENT})
            with urllib.request.urlopen(req, timeout=300) as r:
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        on_bytes(len(chunk))
            if expected and got != expected:
                raise RuntimeError(f"{dest.name}: got {got} bytes, expected {expected}")
            os.replace(tmp, dest)
            return
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, RuntimeError) as e:
            last = e
        if got:
            on_bytes(-got)
        tmp.unlink(missing_ok=True)
        time.sleep(1.5 * (i + 1))
    assert last is not None
    raise last


def write_manifest(root: Path, ref: str, manifest: dict) -> None:
    model, tag = split_ref(ref)
    path = manifest_path(root, model, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest), encoding="utf-8")
    os.replace(tmp, path)


def ref_ready(root: Path, ref: str, manifest: dict) -> bool:
    if not manifest_path(root, *split_ref(ref)).exists():
        return False
    cfg = manifest.get("config") or {}
    items = [cfg, *(manifest.get("layers") or [])]
    for item in items:
        if not isinstance(item, dict) or not item.get("digest"):
            continue
        if not blob_present(blob_path(root, item["digest"]), int(item.get("size") or 0)):
            return False
    return True


def hydrate(
    refs: list[str],
    root: Path | None = None,
    workers: int = 8,
    on_batch=None,
) -> dict:
    root = root or models_dir()
    root.mkdir(parents=True, exist_ok=True)
    blobs, manifests = collect_blobs(refs)
    have = [b for b in blobs if blob_present(blob_path(root, b["digest"]), b["size"])]
    missing = [b for b in blobs if not blob_present(blob_path(root, b["digest"]), b["size"])]
    have_bytes = sum(b["size"] for b in have)
    miss_bytes = sum(b["size"] for b in missing)
    missing.sort(key=lambda b: b["size"], reverse=True)
    print(
        f"[hydrate] {len(refs)} tags, {len(blobs)} unique blobs "
        f"({fmt_bytes(have_bytes)} present, {fmt_bytes(miss_bytes)} to fetch, {workers} workers)",
        flush=True,
    )
    data_root = Path(os.environ.get("MODELINDEX_DATA", str(Path(__file__).resolve().parents[1] / "data")))
    prog = ProgressFile(data_root / "out" / "job_progress.json", job="pull")
    prog.update(
        tags=len(refs), blobs=len(blobs), bytes_done=have_bytes,
        bytes_total=have_bytes + miss_bytes, workers=workers, phase="download",
    )
    prog.start()
    bar = Bar("hydrate", have_bytes + miss_bytes, unit="B", done=have_bytes)
    bar_lock = threading.Lock()
    current = {"digest": ""}
    downloaded = {"n": have_bytes}
    last_pf = [0.0]

    def on_bytes(n: int) -> None:
        downloaded["n"] += n
        now = time.time()
        with bar_lock:
            bar.update(k=n, item=current["digest"][:19])
            if now - last_pf[0] >= 2:
                last_pf[0] = now
                prog.update(
                    bytes_done=downloaded["n"],
                    bytes_total=have_bytes + miss_bytes,
                    digest=current["digest"][:19],
                )

    def one(blob: dict) -> str:
        dest = blob_path(root, blob["digest"])
        if blob_present(dest, blob["size"]):
            return blob["digest"]
        current["digest"] = blob["digest"]
        t0 = time.time()
        download_blob(blob["url"], dest, blob["size"], on_bytes)
        print(f"[hydrate] {blob['digest'][:19]} {fmt_bytes(blob['size'])} in {fmt_dur(time.time() - t0)}", flush=True)
        return blob["digest"]

    pulled_refs: list[str] = []

    def flush_manifests() -> list[str]:
        newly = []
        for ref, manifest in manifests.items():
            if ref in pulled_refs:
                continue
            if ref_ready(root, ref, manifest):
                newly.append(ref)
                continue
            items = [manifest.get("config") or {}, *(manifest.get("layers") or [])]
            if all(
                blob_present(blob_path(root, item["digest"]), int(item.get("size") or 0))
                for item in items
                if isinstance(item, dict) and item.get("digest")
            ):
                write_manifest(root, ref, manifest)
                newly.append(ref)
                print(f"[hydrate] manifest {ref}", flush=True)
        pulled_refs.extend(newly)
        return newly

    flush_manifests()
    if on_batch and pulled_refs:
        on_batch(list(pulled_refs))
    if missing:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            pending = iter(missing)
            while True:
                batch = []
                for _ in range(max(1, workers)):
                    try:
                        batch.append(next(pending))
                    except StopIteration:
                        break
                if not batch:
                    break
                futs = [ex.submit(one, b) for b in batch]
                for fut in as_completed(futs):
                    fut.result()
                flush_manifests()
                if on_batch:
                    on_batch(list(pulled_refs))
    bar.finish()
    ready = [ref for ref, man in manifests.items() if ref_ready(root, ref, man)]
    missing_refs = [r for r in refs if r not in ready]
    if missing_refs:
        prog.close(status="error")
        raise RuntimeError(f"hydrate incomplete: {missing_refs}")
    prog.close(status="done")
    return {
        "refs": ready,
        "blobs": len(blobs),
        "fetched": len(missing),
        "bytes": miss_bytes,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True, help="comma-separated model:tag list")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dir", help="OLLAMA_MODELS directory (default: env or ~/.ollama/models)")
    a = ap.parse_args(argv)
    refs = [m.strip() for m in a.models.split(",") if m.strip()]
    out = hydrate(refs, Path(a.dir) if a.dir else None, workers=a.workers)
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
