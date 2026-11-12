"""The manifest every stage reads and writes: one JSON object per line.

An item is a recording and its notes, ``[onset, offset, midi_pitch]`` in
seconds from the start of that recording. Later stages add fields (the mix's
path and gains, the separated stem, the feature file) rather than rewrite the
ones before, so a single line says where everything about one example came from.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List


def read(path: str | Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path: str | Path, items: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def split(items: Iterable[dict], name: str) -> Iterator[dict]:
    return (it for it in items if it.get("split") == name)
