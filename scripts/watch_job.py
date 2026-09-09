#!/usr/bin/env python3
"""Watch a MODELINDEX job until it finishes, stalls, or dies.

Reads the atomic JSON written by ProgressFile (Volume or local), and/or tails a
Modal `modal run` log. Stdlib only.

    python scripts/watch_job.py --log path/to/terminal.txt
    python scripts/watch_job.py --volume modelindex-data --remote /out/job_progress.json
    python scripts/watch_job.py --progress data/out/job_progress.json --interval 15

Exit codes: 0 done, 1 error/dead, 2 stall (heartbeat alive but phase frozen).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_HINTS = re.compile(
    r"(=== .+ ===|\[vram\]|\[speed\]|\[progress\]|\[done\]|\[skip\]|\[hydrate\]|"
    r"ConflictError|Traceback|RemoteError|App completed|gate.: .PASS)"
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S+00:00"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def load_progress(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def pull_volume(volume: str, remote: str, dest: Path) -> dict | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd = ["modal", "volume", "get", "--force", volume, remote, str(dest)]
    try:
        subprocess.run(cmd, check=False, capture_output=True, text=True, env=env, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return load_progress(dest)


def tail_log(path: Path, last_pos: int) -> tuple[int, list[str], float | None]:
    if not path.exists():
        return last_pos, [], None
    mtime = path.stat().st_mtime
    size = path.stat().st_size
    if size < last_pos:
        last_pos = 0
    hints = []
    with open(path, encoding="utf-8", errors="replace") as f:
        f.seek(last_pos)
        chunk = f.read()
        last_pos = f.tell()
    for line in chunk.splitlines():
        if LOG_HINTS.search(line):
            hints.append(line.rstrip())
    return last_pos, hints, mtime


def summarize_log(path: Path) -> dict:
    model = None
    last_hint = None
    if not path.exists():
        return {"model": None, "last": None, "silence_s": None}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("===") and line.endswith("==="):
                model = line.strip("= ").strip()
            if LOG_HINTS.search(line):
                last_hint = line
    silence = time.time() - path.stat().st_mtime
    return {"model": model, "last": last_hint, "silence_s": round(silence, 1)}


def classify(prog: dict | None, log: dict, stale_s: float, stall_s: float) -> str:
    if prog:
        status = prog.get("status")
        if status == "done":
            return "done"
        if status == "error":
            return "error"
        hb = parse_ts(prog.get("heartbeat_ts"))
        if hb and (utc_now() - hb).total_seconds() > stale_s:
            return "dead"
        phase_elapsed = float(prog.get("phase_elapsed_s") or 0)
        if prog.get("phase") in ("generate", "load") and phase_elapsed > stall_s:
            return "stall"
    if log.get("last") and "ConflictError" in (log.get("last") or ""):
        return "error"
    last = log.get("last") or ""
    # A per-pass "[done] wrote …jsonl" is not the App finishing (smoke/main/kvq8).
    if "App completed" in last or "local entrypoint completed" in last:
        return "done"
    if log.get("silence_s") is not None and log["silence_s"] > stall_s:
        return "stall"
    return "running"


def render(kind: str, prog: dict | None, log: dict) -> str:
    bits = [f"watch={kind}"]
    if prog:
        bits.append(
            f"job={prog.get('job')} status={prog.get('status')} phase={prog.get('phase')} "
            f"model={prog.get('model')} ctx={prog.get('ctx')} prompt={prog.get('prompt')} "
            f"run={prog.get('run')} elapsed={prog.get('elapsed_s')}s "
            f"phase_elapsed={prog.get('phase_elapsed_s')}s "
            f"done={len(prog.get('models_done') or [])}/{prog.get('models_total')}"
        )
    if log.get("model"):
        bits.append(f"log_model={log['model']}")
    if log.get("silence_s") is not None:
        bits.append(f"log_silence={log['silence_s']}s")
    if log.get("last"):
        bits.append("last=" + log["last"][-160:])
    return " | ".join(bits)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--progress", help="local job_progress.json")
    ap.add_argument("--volume", help="Modal Volume name (optional)")
    ap.add_argument("--remote", default="/out/job_progress.json")
    ap.add_argument("--log", help="modal run / terminal log to tail")
    ap.add_argument("--interval", type=float, default=20)
    ap.add_argument("--stale-s", type=float, default=90, help="heartbeat older than this => dead")
    ap.add_argument("--stall-s", type=float, default=180, help="generate/load or log silence => stall")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--fail-on-stall", action="store_true")
    a = ap.parse_args(argv)

    local = Path(a.progress) if a.progress else Path("data/out/job_progress.json")
    log_path = Path(a.log) if a.log else None
    last_pos = 0
    last_kind = None
    stall_announced = False

    while True:
        prog = load_progress(local) if a.progress or local.exists() else None
        if a.volume:
            pulled = pull_volume(a.volume, a.remote, local)
            if pulled:
                prog = pulled
        log = summarize_log(log_path) if log_path else {"model": None, "last": None, "silence_s": None}
        if log_path:
            last_pos, hints, _ = tail_log(log_path, last_pos)
            for h in hints[-8:]:
                print("  " + h, flush=True)
        kind = classify(prog, log, a.stale_s, a.stall_s)
        print(render(kind, prog, log), flush=True)
        if kind == "stall" and not stall_announced:
            stall_announced = True
        if kind != "stall":
            stall_announced = False
        last_kind = kind
        if a.once:
            if kind == "done":
                return 0
            if kind in ("error", "dead"):
                return 1
            if kind == "stall":
                return 2
            return 0
        if kind == "done":
            return 0
        if kind in ("error", "dead"):
            return 1
        if kind == "stall" and a.fail_on_stall:
            return 2
        time.sleep(max(5.0, a.interval))


if __name__ == "__main__":
    raise SystemExit(main())
