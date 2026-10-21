"""Timestamped progress lines for long stages, so a Slurm log shows how far a job has got and when it will finish.

    p = Progress(len(rows), "features")
    for row in rows: ...; p.tick()

prints at most once a minute, and once at the end:
``2026-10-21T14:03:12Z [features] 1200/17117 (7%)  9.8/s  elapsed 2m02s  eta 27m03s``
"""
from __future__ import annotations

import time


def _hms(s: float) -> str:
    s = int(max(s, 0))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


class Progress:
    def __init__(self, total: int, label: str, every_s: float = 60.0, start: int = 0):
        self.total, self.label, self.every_s = total, label, every_s
        self.done, self.start_done = start, start
        self.t0 = self.last = time.time()

    def line(self) -> str:
        el = time.time() - self.t0
        rate = (self.done - self.start_done) / el if el > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else float("nan")
        pct = 100.0 * self.done / self.total if self.total else 100.0
        return (f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [{self.label}] {self.done}/{self.total} ({pct:.0f}%)"
                f"  {rate:.1f}/s  elapsed {_hms(el)}  eta {_hms(eta) if eta == eta else '?'}")

    def tick(self, n: int = 1) -> None:
        self.done += n
        now = time.time()
        if now - self.last >= self.every_s or self.done >= self.total:
            self.last = now
            print(self.line(), flush=True)
