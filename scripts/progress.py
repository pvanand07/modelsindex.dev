"""ASCII progress bar, atomic JSON progress file, heartbeat. Stdlib only."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path


def fmt_dur(seconds: float) -> str:
    if seconds != seconds or seconds < 0:  # NaN
        return "?"
    s = int(round(seconds))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def fmt_bytes(n: float) -> str:
    n = float(n)
    for unit, div in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024.0)):
        if abs(n) >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n:.0f} B"


class Bar:
    def __init__(
        self,
        label: str,
        total: int,
        width: int = 28,
        min_interval: float = 1.0,
        done: int = 0,
        unit: str = "",
    ):
        self.label = label
        self.total = max(int(total), 0)
        self.width = width
        self.min_interval = min_interval
        self.unit = unit
        self.n = max(int(done), 0)
        self._rate_n = 0
        self.t0 = time.perf_counter()
        self._last = 0.0
        self._draw("")

    def update(self, k: int = 1, item: str = "") -> None:
        self.n += k
        self._rate_n += k
        now = time.perf_counter()
        done = self.total > 0 and self.n >= self.total
        if not done and now - self._last < self.min_interval:
            return
        self._last = now
        self._draw(item)

    def finish(self, item: str = "") -> None:
        if self.total:
            self.n = max(self.n, self.total)
        self._draw(item)

    def _fmt_n(self, n: float) -> str:
        return fmt_bytes(n) if self.unit == "B" else f"{int(n)}"

    def _draw(self, item: str) -> None:
        elapsed = time.perf_counter() - self.t0
        if self.total <= 0:
            print(f"[{self.label}] 0/0", flush=True)
            return
        frac = min(1.0, self.n / self.total)
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        rate = self._rate_n / elapsed if elapsed > 0 else 0.0
        remain = (self.total - self.n) / rate if rate > 0 and self.n < self.total else 0.0
        extra = f"  {item}" if item else ""
        rate_s = f"{fmt_bytes(rate)}/s" if self.unit == "B" else f"{rate:.1f}/s"
        print(
            f"[{self.label}] [{bar}] {self._fmt_n(self.n)}/{self._fmt_n(self.total)} "
            f"{100 * frac:5.1f}%  {rate_s}  elapsed {fmt_dur(elapsed)}  eta {fmt_dur(remain)}{extra}",
            flush=True,
        )
        sys.stderr.flush()
        sys.stdout.flush()


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ProgressFile:
    """Atomic JSON snapshot + 5s heartbeat. Watch with scripts/watch_job.py."""

    def __init__(self, path: Path | str | None, job: str = ""):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.state: dict = {
            "job": job,
            "status": "running",
            "heartbeat_ts": _utc(),
            "elapsed_s": 0.0,
            "phase": "start",
            "phase_t0": time.time(),
        }
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.write(emit_log=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, emit_log: bool = False, **kwargs) -> None:
        with self.lock:
            if "phase" in kwargs and kwargs["phase"] != self.state.get("phase"):
                kwargs["phase_t0"] = time.time()
                emit_log = True
            self.state.update(kwargs)
            self._flush(emit_log=emit_log)

    def write(self, emit_log: bool = False) -> None:
        with self.lock:
            self._flush(emit_log=emit_log)

    def close(self, status: str = "done") -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        self.update(emit_log=True, status=status)

    def _loop(self) -> None:
        while not self._stop.wait(5):
            self.write(emit_log=False)

    def _flush(self, emit_log: bool) -> None:
        self.state["heartbeat_ts"] = _utc()
        self.state["elapsed_s"] = round(time.time() - self.t0, 1)
        phase_t0 = self.state.get("phase_t0") or self.t0
        self.state["phase_elapsed_s"] = round(time.time() - phase_t0, 1)
        if not self.path:
            if emit_log:
                print("[progress] " + json.dumps(self.state, default=str), flush=True)
            return
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = json.dumps(self.state, default=str)
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.path)
        if emit_log:
            print("[progress] " + payload, flush=True)
