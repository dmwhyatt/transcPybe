"""Give existing mixes, stems or features new labels, without mixing or separating again.

    python -m boundary.relabel STEMS.jsonl VOCALS.jsonl OUT.jsonl

Each row of STEMS.jsonl (any stage's manifest, from mixes.jsonl on) takes the
notes of the row in VOCALS.jsonl with the same ``id``, moved by the silence that
was put in front of it when it was mixed (``pad_before``). Everything else is
kept, so the audio is reused as it is: a mix's draws do not depend on its notes.
This is how an onset-convention ablation reuses separated stems, the expensive
part.

The mix ids stay as they are, so run ``features`` into a new directory. A
feature file records the notes it was made with, so a mismatched one is never
reused, but in the same directory it would be overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import items as manifest


def relabel(rows, vocals):
    by_id = {v["id"]: v for v in vocals}
    missing = [r["id"] for r in rows if r["id"] not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} rows have no labels in the vocals manifest, e.g. {missing[0]}")
    out = []
    for r in rows:
        v, pad = by_id[r["id"]], r.get("pad_before", 0.0)
        out.append(dict(r, onsets=v.get("onsets"), relabelled_from=r.get("onsets"),
                        notes=[[round(a + pad, 4), round(b + pad, 4), p] for a, b, p in v["notes"]]))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", type=Path)
    ap.add_argument("vocals", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args(argv)
    n = manifest.write(args.out, relabel(manifest.read(args.rows), manifest.read(args.vocals)))
    print(f"{n} rows relabelled -> {args.out}")


if __name__ == "__main__":
    main()
