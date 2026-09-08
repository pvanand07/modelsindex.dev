#!/usr/bin/env python3
"""
Fetch and parse a GGUF header without downloading the weights.

Works on a URL (HTTP Range request, follows the registry's 307 redirect) or a local file.
Returns the metadata, the full tensor table, and derived fields needed for VRAM/speed math:
param count, per-tensor-group byte sums, MoE active bytes, KV bytes per token.

Stdlib only.

    python scripts/gguf_header.py https://registry.ollama.ai/v2/library/llama3.2/blobs/sha256:...
    python scripts/gguf_header.py path/to/model.gguf --json
    python scripts/gguf_header.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import urllib.request

USER_AGENT = "modelindex-crawler/0.1 (+https://github.com/local/modelindex)"
FIRST_FETCH = 8 * 1024 * 1024
MAX_FETCH = 512 * 1024 * 1024
ARRAY_SUMMARY_THRESHOLD = 64  # arrays longer than this are stored as {"__len__": n}

SELFTEST_URL = (
    "https://registry.ollama.ai/v2/library/llama3.2/blobs/"
    "sha256:dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff"
)
SELFTEST_EXPECT = {"param_count": 3212749888, "n_layer": 28, "n_head_kv": 8, "key_length": 128}


class NeedMoreBytes(Exception):
    """Raised when the buffer ends before the header does."""


# --------------------------------------------------------------------------- binary reader
class _Reader:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.p = 0

    def rd(self, fmt: str):
        fmt = "<" + fmt
        n = struct.calcsize(fmt)
        if self.p + n > len(self.buf):
            raise NeedMoreBytes
        v = struct.unpack_from(fmt, self.buf, self.p)[0]
        self.p += n
        return v

    def rstr(self) -> str:
        n = self.rd("Q")
        if self.p + n > len(self.buf):
            raise NeedMoreBytes
        s = self.buf[self.p : self.p + n]
        self.p += n
        return s.decode("utf-8", "replace")


_SCALAR_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_STRING, _ARRAY = 8, 9


def _read_value(r: _Reader, t: int, summarize: bool):
    if t == _STRING:
        return r.rstr()
    if t == _ARRAY:
        et = r.rd("I")
        n = r.rd("Q")
        if summarize and n > ARRAY_SUMMARY_THRESHOLD:
            for _ in range(n):  # must still consume the bytes
                _read_value(r, et, False)
            return {"__len__": n}
        return [_read_value(r, et, False) for _ in range(n)]
    return r.rd(_SCALAR_FMT[t])


# --------------------------------------------------------------------------- parsing
def parse_gguf(buf: bytes, summarize_arrays: bool = True) -> dict:
    """Parse header + tensor table from the first bytes of a GGUF file.

    Raises NeedMoreBytes if `buf` is too short. Returns a dict with keys:
    version, n_tensors, n_kv, metadata, tensors, header_end, data_start.
    """
    if buf[:4] != b"GGUF":
        raise ValueError("not a GGUF file (bad magic)")
    r = _Reader(buf)
    r.p = 4
    version = r.rd("I")
    n_tensors = r.rd("Q")
    n_kv = r.rd("Q")
    meta: dict = {}
    for _ in range(n_kv):
        k = r.rstr()
        t = r.rd("I")
        meta[k] = _read_value(r, t, summarize_arrays)
    tensors = []
    for _ in range(n_tensors):
        name = r.rstr()
        nd = r.rd("I")
        dims = [r.rd("Q") for _ in range(nd)]
        ttype = r.rd("I")
        off = r.rd("Q")
        tensors.append({"name": name, "dims": dims, "type": ttype, "offset": off})
    header_end = r.p
    align = int(meta.get("general.alignment", 32) or 32)
    data_start = math.ceil(header_end / align) * align
    return {
        "version": version,
        "n_tensors": n_tensors,
        "n_kv": n_kv,
        "metadata": meta,
        "tensors": tensors,
        "header_end": header_end,
        "data_start": data_start,
    }


# --------------------------------------------------------------------------- fetching
def fetch_range(url: str, start: int, end: int, timeout: int = 120) -> tuple[bytes, int | None]:
    """GET bytes [start, end] of `url`. Returns (bytes, total_size_or_None)."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        total = None
        cr = resp.headers.get("Content-Range")
        if cr and "/" in cr:
            try:
                total = int(cr.rsplit("/", 1)[1])
            except ValueError:
                total = None
        if resp.status == 200 and total is None:
            total = len(data)  # server ignored Range and sent the whole file
    return data, total


def load_header(source: str) -> tuple[dict, int, int]:
    """Load and parse a header from a URL or local path.

    Returns (parsed, file_size, bytes_read).
    """
    if source.startswith("http://") or source.startswith("https://"):
        size = FIRST_FETCH
        while True:
            buf, total = fetch_range(source, 0, size - 1)
            try:
                parsed = parse_gguf(buf)
                return parsed, (total or len(buf)), len(buf)
            except NeedMoreBytes:
                if size >= MAX_FETCH or (total and size >= total):
                    raise RuntimeError(f"header longer than {size} bytes; giving up")
                size *= 4
    else:
        file_size = os.path.getsize(source)
        size = FIRST_FETCH
        while True:
            with open(source, "rb") as f:
                buf = f.read(size)
            try:
                return parse_gguf(buf), file_size, len(buf)
            except NeedMoreBytes:
                if size >= file_size:
                    raise
                size *= 4


# --------------------------------------------------------------------------- derived fields
def _scalar_or_list_sum(v, n_layer: int) -> int:
    """head_count_kv may be a per-layer list. Return the sum over layers."""
    if isinstance(v, list):
        return int(sum(v))
    if isinstance(v, dict) and "__len__" in v:
        return 0  # summarized array; caller falls back
    return int(v) * n_layer


def _scalar_or_list_max(v) -> int:
    if isinstance(v, list):
        return int(max(v))
    if isinstance(v, dict):
        return 0
    return int(v)


def tensor_group(name: str) -> str:
    if name.startswith("token_embd"):
        return "token_embd"
    if name.startswith("v.") or name.startswith("mm.") or name.startswith("a."):
        return "vision"  # embedded vision/audio tower (gemma3, llava-style), not used in text decode
    if name.startswith("output"):
        return "output"
    if "_exps." in name:
        return "ffn_expert"
    if ".attn_" in name:
        return "attn"
    if ".ffn_" in name:
        return "ffn_dense"
    return "other"


def tensor_sizes_from_offsets(tensors: list[dict], data_start: int, file_size: int) -> list[int]:
    """Byte size per tensor derived from consecutive offsets (exact up to alignment padding)."""
    order = sorted(range(len(tensors)), key=lambda i: tensors[i]["offset"])
    data_len = file_size - data_start
    sizes = [0] * len(tensors)
    for k, i in enumerate(order):
        start = tensors[i]["offset"]
        end = tensors[order[k + 1]]["offset"] if k + 1 < len(order) else data_len
        sizes[i] = max(0, end - start)
    return sizes


def summarize(parsed: dict, file_size: int) -> dict:
    meta = parsed["metadata"]
    tensors = parsed["tensors"]
    arch = meta.get("general.architecture", "unknown")

    def m(key, default=None):
        return meta.get(f"{arch}.{key}", default)

    n_layer = int(m("block_count", 0) or 0)
    n_embd = int(m("embedding_length", 0) or 0)
    n_head_raw = m("attention.head_count", 0)
    n_head_kv_raw = m("attention.head_count_kv", n_head_raw)
    n_head = _scalar_or_list_max(n_head_raw)
    n_head_kv = _scalar_or_list_max(n_head_kv_raw)
    kv_heads_sum = _scalar_or_list_sum(n_head_kv_raw, n_layer) or n_head_kv * n_layer
    head_dim_default = (n_embd // n_head) if n_head else 0
    key_length = int(m("attention.key_length", head_dim_default) or head_dim_default)
    value_length = int(m("attention.value_length", head_dim_default) or head_dim_default)

    vocab = m("vocab_size")
    if not vocab:
        toks = meta.get("tokenizer.ggml.tokens")
        if isinstance(toks, dict):
            vocab = toks.get("__len__")
        elif isinstance(toks, list):
            vocab = len(toks)
    if not vocab:
        for t in tensors:
            if t["name"].startswith("token_embd") and len(t["dims"]) == 2:
                vocab = t["dims"][1]
                break

    sizes = tensor_sizes_from_offsets(tensors, parsed["data_start"], file_size)
    groups: dict[str, int] = {}
    for t, s in zip(tensors, sizes):
        g = tensor_group(t["name"])
        groups[g] = groups.get(g, 0) + s
    param_count = sum(math.prod(t["dims"]) for t in tensors)

    expert_count = int(m("expert_count", 0) or 0)
    expert_used = int(m("expert_used_count", 0) or 0)
    expert_bytes = groups.get("ffn_expert", 0)
    if expert_count and expert_used and expert_bytes:
        active_bytes = (file_size - expert_bytes) + expert_bytes * expert_used / expert_count
    else:
        active_bytes = file_size
    # active params: scale expert params the same way
    expert_params = sum(math.prod(t["dims"]) for t in tensors if tensor_group(t["name"]) == "ffn_expert")
    if expert_count and expert_used and expert_params:
        active_params = (param_count - expert_params) + expert_params * expert_used / expert_count
    else:
        active_params = param_count

    kv_bytes_per_token_f16 = kv_heads_sum * (key_length + value_length) * 2

    return {
        "arch": arch,
        "name": meta.get("general.name"),
        "file_type": meta.get("general.file_type"),
        "size_label": meta.get("general.size_label"),
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "kv_heads_sum": kv_heads_sum,
        "key_length": key_length,
        "value_length": value_length,
        "context_length": int(m("context_length", 0) or 0),
        "vocab_size": int(vocab or 0),
        "expert_count": expert_count,
        "expert_used_count": expert_used,
        "sliding_window": int(m("attention.sliding_window", 0) or 0),
        "param_count": int(param_count),
        "active_params": int(active_params),
        "tensor_count": len(tensors),
        "weight_bytes": int(file_size),
        "data_start": parsed["data_start"],
        "tensor_groups_bytes": groups,
        "expert_bytes": int(expert_bytes),
        "active_weight_bytes": int(active_bytes),
        "kv_bytes_per_token_f16": int(kv_bytes_per_token_f16),
    }


def header_summary(source: str) -> dict:
    parsed, file_size, nread = load_header(source)
    out = summarize(parsed, file_size)
    out["header_bytes"] = parsed["header_end"]
    out["bytes_fetched"] = nread
    return out


# --------------------------------------------------------------------------- CLI
def _selftest() -> int:
    s = header_summary(SELFTEST_URL)
    ok = True
    for k, v in SELFTEST_EXPECT.items():
        good = s.get(k) == v
        ok &= good
        print(f"  {k:14s} expected {v:>12}  got {s.get(k)!s:>12}  {'OK' if good else 'FAIL'}")
    total_groups = sum(s["tensor_groups_bytes"].values())
    rel = abs(total_groups - (s["weight_bytes"] - s["data_start"])) / s["weight_bytes"]
    good = rel < 1e-4
    ok &= good
    print(f"  group byte sum vs data length: rel err {rel:.2e}  {'OK' if good else 'FAIL'}")
    good = s["bytes_fetched"] < 32 * 1024 * 1024
    ok &= good
    print(f"  bytes fetched {s['bytes_fetched']:,} (header {s['header_bytes']:,})  {'OK' if good else 'FAIL'}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="URL or local .gguf path")
    ap.add_argument("--json", action="store_true", help="print full summary as JSON")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if not a.source:
        ap.error("source required")
    s = header_summary(a.source)
    if a.json:
        print(json.dumps(s, indent=2))
    else:
        for k in ("arch", "name", "n_layer", "n_head", "n_head_kv", "key_length", "value_length",
                  "context_length", "vocab_size", "expert_count", "expert_used_count", "sliding_window",
                  "param_count", "active_params", "weight_bytes", "active_weight_bytes",
                  "kv_bytes_per_token_f16", "header_bytes", "bytes_fetched"):
            print(f"{k:24s} {s[k]}")
        print("tensor_groups_bytes      " + json.dumps(s["tensor_groups_bytes"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
